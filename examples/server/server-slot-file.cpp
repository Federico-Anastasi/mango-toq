#include "server-slot-file.h"
#include "server-common.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#define XXH_INLINE_ALL
#include "xxhash/xxhash.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static constexpr uint32_t SLOT_FILE_MAGIC   = 0x746c736du; // "mslt"
static constexpr uint32_t SLOT_FILE_VERSION = 1;
static constexpr size_t   SLOT_FILE_FOOTER  = 8 + 8 + 8 + 4 + 4;

// Disk cache thresholds, in prompt tokens to reread (~600 tokens/s on the 27B).
static constexpr size_t  DISK_MIN_GAIN    = 256;  // a file is loaded if it reuses at least this many more tokens than the slot
static constexpr size_t  DISK_MIN_SAVE    = 1024; // a state is saved if rebuilding it would cost at least this much
static constexpr size_t  DISK_MIN_BASE    = 1024; // a shorter common prefix is not worth a file
static constexpr int64_t DISK_IDLE_SAVE_S = 30;   // slot idle for long enough: save it
static constexpr size_t  DISK_MIN_FREE    = 2ull << 30; // at least 2 GiB always stay free on the disk

static uint64_t tokens_hash(const llama_token * t, size_t n) {
    return XXH64(t, n * sizeof(llama_token), 0);
}

static uint64_t tokens_hash(const server_tokens & t) {
    const llama_tokens v = t.tokens_data();
    return tokens_hash(v.data(), v.size());
}

// The base file name: depends only on its tokens.
static std::string base_file_name(uint64_t hash) {
    char name[40];
    snprintf(name, sizeof(name), "base-%016llx.slot", (unsigned long long) hash);
    return name;
}

// How many tokens of the prompt are the common prefix to keep as the base (system prompt and tools:
// `base_text` is the text that precedes the first user message), 0 if the prompt has none worth
// a file or if nothing is left to read after it.
static size_t base_length(llama_context * ctx, const std::string & base_text, const llama_tokens & ptok) {
    if (base_text.empty()) {
        return 0;
    }
    const llama_tokens base = common_tokenize(ctx, base_text, true, true);
    size_t n = 0;
    while (n < base.size() && n < ptok.size() && base[n] == ptok[n] && ptok[n] != LLAMA_TOKEN_NULL) {
        ++n;
    }
    return (n < DISK_MIN_BASE || n + 1 >= ptok.size()) ? 0 : n;
}

static int64_t file_time_s(const fs::file_time_type & t) {
    return std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
}

static int64_t now_s() {
    return file_time_s(fs::file_time_type::clock::now());
}

uint64_t slot_file_model_fingerprint(const gpt_params & params) {
    XXH64_state_t h;
    XXH64_reset(&h, 0);
    auto add_file = [&](const std::string & path) {
        const uint64_t size = fs::file_size(path);
        XXH64_update(&h, &size, sizeof(size));
        std::ifstream f(path, std::ios::binary);
        const uint64_t chunk = std::min<uint64_t>(size, 4ull << 20);
        std::vector<char> buf(chunk);
        f.read(buf.data(), chunk);
        f.seekg(size - chunk);
        XXH64_update(&h, buf.data(), chunk);
        f.read(buf.data(), chunk);
        if (!f) {
            throw std::runtime_error("cannot read " + path + " for the state fingerprint");
        }
        XXH64_update(&h, buf.data(), chunk);
    };
    add_file(params.model);
    for (const auto & la : params.lora_adapters) {
        add_file(la.path);
        XXH64_update(&h, &la.scale, sizeof(la.scale));
    }
    const std::string kv = params.cache_type_k + "/" + params.cache_type_v;
    XXH64_update(&h, kv.data(), kv.size());
    XXH64_update(&h, &SLOT_FILE_VERSION, sizeof(SLOT_FILE_VERSION));
    return XXH64_digest(&h);
}

struct slot_file_ckpt_meta {
    server_prompt_checkpoint ckpt; // without data
    uint64_t offset = 0;
    uint64_t size   = 0;
    uint64_t hash   = 0;
};

struct slot_file_meta {
    uint64_t      model = 0;
    bool          base  = false;
    uint64_t      state_size = 0;
    uint64_t      state_hash = 0;
    server_tokens tokens;
    int32_t       n_kept_prompt = 0;
    int32_t       n_discarded_prompt = 0;
    std::vector<slot_file_ckpt_meta> ckpts;
    uint64_t      mtp_offset = 0; // cache of the MTP layer, 0 bytes if the file does not carry it,
    uint64_t      mtp_size   = 0; // then the hidden state of the last position (mtp_hidden floats)
    uint64_t      mtp_hidden = 0;
    uint64_t      mtp_hash   = 0; // of both
};

// The context of the slot's MTP layer (it has its own cache), nullptr with MTP off.
static llama_context * slot_mtp_ctx(const server_slot & slot) {
    return slot.spec ? common_speculative_get_companion_ctx(slot.spec) : nullptr;
}

// Footer and metadata, with all the consistency checks that do not require reading the state.
static bool slot_file_read_meta(const std::string & path, slot_file_meta & meta, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "file not found";
        return false;
    }
    std::error_code ec;
    const uint64_t size = fs::file_size(path, ec);
    if (ec || size < SLOT_FILE_FOOTER) {
        err = "file truncated";
        return false;
    }
    uint64_t meta_off = 0, meta_size = 0, meta_hash = 0;
    uint32_t version = 0, magic = 0;
    f.seekg(size - SLOT_FILE_FOOTER);
    f.read((char *) &meta_off, 8).read((char *) &meta_size, 8).read((char *) &meta_hash, 8);
    f.read((char *) &version, 4).read((char *) &magic, 4);
    if (!f || magic != SLOT_FILE_MAGIC) {
        err = "not a state file (or truncated)";
        return false;
    }
    if (version != SLOT_FILE_VERSION) {
        err = "format version " + std::to_string(version) + ", expected " + std::to_string(SLOT_FILE_VERSION);
        return false;
    }
    if (meta_off + meta_size + SLOT_FILE_FOOTER != size) {
        err = "footer inconsistent with the file size";
        return false;
    }
    std::string m(meta_size, '\0');
    f.seekg(meta_off);
    f.read(m.data(), meta_size);
    if (!f || XXH64(m.data(), m.size(), 0) != meta_hash) {
        err = "corrupted metadata";
        return false;
    }
    try {
        const json j = json::parse(m);
        meta.model              = j.at("model").get<uint64_t>();
        meta.base               = j.at("base").get<bool>();
        meta.state_size         = j.at("state_size").get<uint64_t>();
        meta.state_hash         = j.at("state_hash").get<uint64_t>();
        meta.n_kept_prompt      = j.at("n_kept_prompt").get<int32_t>();
        meta.n_discarded_prompt = j.at("n_discarded_prompt").get<int32_t>();
        meta.tokens.from_json(j.at("tokens"));
        uint64_t end = meta.state_size;
        for (const auto & c : j.at("checkpoints")) {
            slot_file_ckpt_meta cm;
            cm.ckpt.pos_min    = c.at("pos_min").get<llama_pos>();
            cm.ckpt.pos_max    = c.at("pos_max").get<llama_pos>();
            cm.ckpt.n_tokens   = c.at("n_tokens").get<int64_t>();
            cm.ckpt.prompt_end = c.at("prompt_end").get<bool>();
            cm.offset          = c.at("offset").get<uint64_t>();
            cm.size            = c.at("size").get<uint64_t>();
            cm.hash            = c.at("hash").get<uint64_t>();
            if (cm.offset != end || cm.size == 0) {
                err = "checkpoint out of place in the metadata";
                return false;
            }
            end += cm.size;
            meta.ckpts.push_back(std::move(cm));
        }
        if (j.contains("mtp")) {
            const json & mj = j.at("mtp");
            meta.mtp_offset = mj.at("offset").get<uint64_t>();
            meta.mtp_size   = mj.at("size").get<uint64_t>();
            meta.mtp_hidden = mj.at("hidden").get<uint64_t>();
            meta.mtp_hash   = mj.at("hash").get<uint64_t>();
            if (meta.mtp_offset != end || meta.mtp_size == 0) {
                err = "MTP cache out of place in the metadata";
                return false;
            }
            end += meta.mtp_size + meta.mtp_hidden * sizeof(float);
        }
        if (end != meta_off) {
            err = "the file parts do not match the metadata";
            return false;
        }
    } catch (const std::exception & e) {
        err = std::string("unreadable metadata: ") + e.what();
        return false;
    }
    return true;
}

// The checkpoints that go into the file: the end-of-prompt one (the next turn renders the answer without the
// reasoning and diverges there) and the last one (if the next turn keeps the answer and diverges towards
// the end, for example in a rewritten tool call). Those in between would only serve prompts that
// diverge earlier, which a saved conversation never sees again: with 32 checkpoints of 150 MB
// on the 27B the file would triple.
static std::vector<const server_prompt_checkpoint *> slot_file_ckpts(const server_slot & slot, bool base) {
    std::vector<const server_prompt_checkpoint *> keep;
    if (base) {
        return keep;
    }
    const server_prompt_checkpoint * prompt_end = nullptr;
    const server_prompt_checkpoint * last = nullptr;
    for (const auto & c : slot.server_cached_prompt.checkpoints) {
        if (c.n_tokens > (int64_t) slot.cache_tokens.size()) {
            continue;
        }
        if (c.prompt_end) {
            prompt_end = &c;
        }
        if (!last || c.n_tokens > last->n_tokens) {
            last = &c;
        }
    }
    if (prompt_end) {
        keep.push_back(prompt_end);
    }
    if (last && last != prompt_end) {
        keep.push_back(last);
    }
    return keep;
}

slot_file_result slot_file_save(llama_context * ctx, const server_slot & slot, const std::string & path,
                                uint64_t model_fp, bool base) {
    const int64_t t0 = ggml_time_us();
    slot_file_result res;
    const std::string tmp = path + ".tmp";
    auto fail = [&](const std::string & why) {
        std::error_code ec;
        fs::remove(tmp, ec);
        res.error = why;
        return res;
    };

    const auto keep = slot_file_ckpts(slot, base);

    const llama_tokens toks = slot.cache_tokens.tokens_data();
    uint64_t state_hash = 0;
    const size_t state_size = llama_state_seq_save_file_hashed(ctx, tmp.c_str(), slot.id, toks.data(), toks.size(), &state_hash);
    if (state_size == 0) {
        return fail("writing the state failed (reason in the previous log line)");
    }

    std::ofstream f(tmp, std::ios::binary | std::ios::app);
    json ckj = json::array();
    uint64_t off = state_size;
    for (const auto * c : keep) {
        GGML_ASSERT(!c->data.empty()); // checkpoint spill is excluded at startup
        f.write((const char *) c->data.data(), c->data.size());
        ckj.push_back({
            { "pos_min",    c->pos_min },
            { "pos_max",    c->pos_max },
            { "n_tokens",   c->n_tokens },
            { "prompt_end", c->prompt_end },
            { "offset",     off },
            { "size",       (uint64_t) c->data.size() },
            { "hash",       (uint64_t) XXH64(c->data.data(), c->data.size(), 0) },
        });
        off += c->data.size();
    }
    json mtpj;
    if (llama_context * ctx_mtp = slot_mtp_ctx(slot)) {
        const size_t kv_size = llama_state_seq_get_size(ctx_mtp, slot.id, 0);
        const std::vector<float> hidden = common_speculative_get_sequence_hidden(slot.spec, slot.id);
        std::vector<uint8_t> buf(kv_size + hidden.size() * sizeof(float));
        if (kv_size == 0 || llama_state_seq_get_data(ctx_mtp, buf.data(), kv_size, slot.id, 0) != kv_size) {
            return fail("reading the MTP cache failed");
        }
        memcpy(buf.data() + kv_size, hidden.data(), hidden.size() * sizeof(float));
        f.write((const char *) buf.data(), buf.size());
        mtpj = { { "offset", off }, { "size", (uint64_t) kv_size }, { "hidden", (uint64_t) hidden.size() },
                 { "hash", (uint64_t) XXH64(buf.data(), buf.size(), 0) } };
        off += buf.size();
    }
    json mj = json{
        { "model",              model_fp },
        { "base",               base },
        { "state_size",         (uint64_t) state_size },
        { "state_hash",         state_hash },
        { "n_kept_prompt",      slot.n_kept_prompt },
        { "n_discarded_prompt", slot.n_discarded_prompt },
        { "tokens",             slot.cache_tokens.to_json() },
        { "checkpoints",        ckj },
    };
    if (!mtpj.is_null()) {
        mj["mtp"] = mtpj;
    }
    const std::string m = mj.dump();
    const uint64_t meta_off = off, meta_size = m.size(), meta_hash = XXH64(m.data(), m.size(), 0);
    f.write(m.data(), m.size());
    f.write((const char *) &meta_off, 8).write((const char *) &meta_size, 8).write((const char *) &meta_hash, 8);
    f.write((const char *) &SLOT_FILE_VERSION, 4).write((const char *) &SLOT_FILE_MAGIC, 4);
    f.close();
    if (!f) {
        return fail("writing the checkpoints or the metadata failed (disk full?)");
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        return fail("rename of " + tmp + " failed: " + ec.message());
    }
    res.ok       = true;
    res.n_tokens = toks.size();
    res.n_bytes  = meta_off + meta_size + SLOT_FILE_FOOTER;
    res.ms       = (ggml_time_us() - t0) / 1000.0;
    return res;
}

slot_file_result slot_file_load(llama_context * ctx, server_slot & slot, const std::string & path, uint64_t model_fp) {
    const int64_t t0 = ggml_time_us();
    slot_file_result res;

    // the slot is emptied first: if loading stops halfway, nothing old or new is left
    llama_context * ctx_mtp = slot_mtp_ctx(slot);
    auto wipe = [&]() {
        if (slot.spec) {
            // also the MTP cache and the hidden state of the last position: they belong to the other conversation
            common_speculative_clear_sequence_kv(slot.spec, ctx, slot.id);
        }
        llama_kv_cache_seq_rm(ctx, slot.id, -1, -1);
        slot.cache_tokens.keep_first(0);
        slot.server_cached_prompt.checkpoints.clear();
        slot.checkpoint_pos = -1;
        slot.n_kept_prompt = 0;
        slot.n_discarded_prompt = 0;
    };
    wipe();

    slot_file_meta meta;
    if (!slot_file_read_meta(path, meta, res.error)) {
        return res;
    }
    if (meta.model != model_fp) {
        res.error = "the file comes from another model or another cache configuration";
        return res;
    }
    // Whether images are read is decided by the running engine (--mmproj), not by the file: a memory saved
    // without the encoder would take the images away from it and the first image would make it crash. Conversely,
    // images in the file and an engine without the encoder are not reloaded.
    if (meta.tokens.has_mtmd_data() && slot.mctx == nullptr) {
        res.error = "the conversation has images, but the engine was started without the image encoder";
        return res;
    }
    meta.tokens.has_mtmd = slot.mctx != nullptr;

    std::list<server_prompt_checkpoint> ckpts;
    {
        std::ifstream f(path, std::ios::binary);
        for (auto & cm : meta.ckpts) {
            server_prompt_checkpoint c = cm.ckpt;
            c.data.resize(cm.size);
            f.seekg(cm.offset);
            f.read((char *) c.data.data(), cm.size);
            if (!f || XXH64(c.data.data(), c.data.size(), 0) != cm.hash) {
                res.error = "corrupted checkpoint (pos " + std::to_string(cm.ckpt.pos_max) + ")";
                return res;
            }
            ckpts.push_back(std::move(c));
        }
    }

    llama_tokens toks(meta.tokens.size());
    size_t n_tok = 0;
    uint64_t hash = 0;
    const size_t nread = llama_state_seq_load_file_hashed(ctx, path.c_str(), slot.id, toks.data(), toks.size(), &n_tok, &hash);
    if (nread == 0) {
        wipe();
        res.error = "the state does not load (reason in the previous log line)";
        return res;
    }
    toks.resize(n_tok);
    if (nread != meta.state_size || hash != meta.state_hash || toks != meta.tokens.tokens_data()) {
        wipe();
        res.error = "corrupted state: the hash does not match";
        return res;
    }

    // The MTP cache only improves the drafts: if it is missing or does not load, the MTP layer warms up on the new
    // tokens and the answer does not change (drafts are verified), so the main state stays.
    if (ctx_mtp && meta.mtp_size == 0) {
        SRV_INF("%s: the file has no MTP cache: drafts restart from the new tokens\n", path.c_str());
    } else if (ctx_mtp) {
        std::vector<uint8_t> buf(meta.mtp_size + meta.mtp_hidden * sizeof(float));
        std::ifstream f(path, std::ios::binary);
        f.seekg(meta.mtp_offset);
        f.read((char *) buf.data(), buf.size());
        std::vector<float> hidden(meta.mtp_hidden);
        memcpy(hidden.data(), buf.data() + meta.mtp_size, hidden.size() * sizeof(float));
        if (!f || XXH64(buf.data(), buf.size(), 0) != meta.mtp_hash ||
            llama_state_seq_set_data(ctx_mtp, buf.data(), meta.mtp_size, slot.id, 0) != meta.mtp_size ||
            (!hidden.empty() && !common_speculative_set_sequence_hidden(slot.spec, slot.id, hidden))) {
            SRV_WRN("%s: MTP cache not loaded: drafts restart from the new tokens\n", path.c_str());
            common_speculative_clear_sequence_hidden(slot.spec, slot.id);
            llama_kv_cache_seq_rm(ctx_mtp, slot.id, -1, -1);
        }
    }

    slot.cache_tokens = std::move(meta.tokens);
    slot.server_cached_prompt.checkpoints = std::move(ckpts);
    slot.checkpoint_pos = slot.server_cached_prompt.checkpoints.empty() ? -1 : slot.server_cached_prompt.checkpoints.back().pos_max;
    slot.n_kept_prompt = meta.n_kept_prompt;
    slot.n_discarded_prompt = meta.n_discarded_prompt;

    res.ok       = true;
    res.n_tokens = n_tok;
    res.n_bytes  = fs::file_size(path);
    res.ms       = (ggml_time_us() - t0) / 1000.0;
    return res;
}

// ---- disk cache ----

// How many tokens of `prompt` are reused starting from a state with these tokens and checkpoints: the whole
// state if it is a prefix of the prompt (at least one token must be read to get the logits), otherwise the
// most advanced checkpoint inside the common prefix. Recurrent state: nothing in between.
static size_t reusable(const server_tokens & state, const std::vector<int64_t> & ckpt_n_tokens, const server_tokens & prompt) {
    if (state.empty() || prompt.empty()) {
        return 0;
    }
    const size_t limit = std::min(state.get_common_prefix_exact(prompt), prompt.size() - 1);
    if (state.size() <= limit) {
        return state.size();
    }
    size_t best = 0;
    for (const int64_t n : ckpt_n_tokens) {
        if ((size_t) n <= limit) {
            best = std::max(best, (size_t) n);
        }
    }
    return best;
}

static std::vector<int64_t> slot_ckpt_n_tokens(const server_slot & slot) {
    std::vector<int64_t> out;
    for (const auto & c : slot.server_cached_prompt.checkpoints) {
        out.push_back(c.n_tokens);
    }
    return out;
}

slot_disk_cache::slot_disk_cache(std::string dir, size_t limit_bytes, int64_t max_age_s, uint64_t model_fp)
    : dir(std::move(dir)), limit_bytes(limit_bytes), max_age_s(max_age_s), model_fp(model_fp) {
    fs::create_directories(this->dir);
}

// Index entry from the file's metadata (without reading the state).
static bool read_entry(const std::string & path, uint64_t model_fp, slot_disk_entry & e, std::string & err) {
    slot_file_meta meta;
    if (!slot_file_read_meta(path, meta, err)) {
        return false;
    }
    if (meta.model != model_fp) {
        err = "comes from another model or another cache configuration";
        return false;
    }
    e.path   = path;
    e.tokens = std::move(meta.tokens);
    e.ckpt_n_tokens.clear();
    for (const auto & c : meta.ckpts) {
        e.ckpt_n_tokens.push_back(c.ckpt.n_tokens);
    }
    e.base   = meta.base;
    e.bytes  = fs::file_size(path);
    e.t_used = file_time_s(fs::last_write_time(path));
    return true;
}

void slot_disk_cache::scan() {
    entries.clear();
    const int64_t now = now_s();
    for (const auto & de : fs::directory_iterator(dir)) {
        if (!de.is_regular_file()) {
            continue;
        }
        const std::string path = de.path().string();
        const std::string ext  = de.path().extension().string();
        std::error_code ec;
        if (ext == ".tmp") {
            SRV_WRN("disk cache: removing %s, half written (shutdown during a save)\n", path.c_str());
            fs::remove(path, ec);
            continue;
        }
        if (ext != ".slot") {
            continue;
        }
        slot_disk_entry e;
        std::string err;
        if (!read_entry(path, model_fp, e, err)) {
            SRV_WRN("disk cache: removing %s: %s\n", path.c_str(), err.c_str());
            fs::remove(path, ec);
            continue;
        }
        if (now - e.t_used > max_age_s) {
            SRV_INF("disk cache: removing %s, unused for %lld hours\n", path.c_str(), (long long) ((now - e.t_used) / 3600));
            fs::remove(path, ec);
            continue;
        }
        entries.push_back(std::move(e));
    }
    SRV_INF("disk cache: %zu states on disk, %.2f GiB (cap %.2f GiB, %lld hours) in %s\n", entries.size(),
            total_bytes() / 1073741824.0, limit_bytes / 1073741824.0, (long long) (max_age_s / 3600), dir.c_str());
}

size_t slot_disk_cache::total_bytes() const {
    size_t s = 0;
    for (const auto & e : entries) {
        s += e.bytes;
    }
    return s;
}

void slot_disk_cache::drop(size_t i, const char * why) {
    SRV_INF("disk cache: removing %s (%zu tokens, %.2f GiB): %s\n", entries[i].path.c_str(), entries[i].tokens.size(),
            entries[i].bytes / 1073741824.0, why);
    std::error_code ec;
    fs::remove(entries[i].path, ec);
    entries.erase(entries.begin() + i);
}

void slot_disk_cache::expire() {
    const int64_t now = now_s();
    for (size_t i = entries.size(); i-- > 0;) {
        if (now - entries[i].t_used > max_age_s) {
            drop(i, "unused for longer than the age limit");
        }
    }
}

// Makes room for a state of `need` bytes: within the cap and leaving DISK_MIN_FREE free on the disk. It first
// checks that removing everything removable would make it fit (otherwise it removes nothing: the base was once
// lost for a state that then did not fit anyway), then removes the least recently used conversations and
// only after that the bases, which weigh little and serve every new conversation.
bool slot_disk_cache::make_room(size_t need, const std::string & keep) {
    expire();
    std::error_code ec;
    const auto sp = fs::space(dir, ec);
    if (ec) {
        SRV_ERR("disk cache: free space on %s unreadable: %s\n", dir.c_str(), ec.message().c_str());
        return false;
    }
    size_t kept = 0, freeable = 0;
    for (const auto & e : entries) {
        (e.path == keep ? kept : freeable) += e.bytes;
    }
    if (kept + need > limit_bytes || sp.available + freeable < need + DISK_MIN_FREE) {
        SRV_WRN("disk cache: a %.2f GiB state does not fit (cap %.2f GiB, %.2f GiB free on the disk, %.0f always stay free): not saved\n",
                need / 1073741824.0, limit_bytes / 1073741824.0, sp.available / 1073741824.0, DISK_MIN_FREE / 1073741824.0);
        return false;
    }
    size_t available = sp.available;
    while (total_bytes() + need > limit_bytes || available < need + DISK_MIN_FREE) {
        size_t victim = entries.size();
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto & e = entries[i];
            if (e.path == keep) {
                continue;
            }
            if (victim == entries.size()) {
                victim = i;
                continue;
            }
            const auto & v = entries[victim];
            if (e.base != v.base ? !e.base : e.t_used < v.t_used) {
                victim = i;
            }
        }
        available += entries[victim].bytes;
        drop(victim, total_bytes() + need > limit_bytes ? "space cap" : "disk almost full");
    }
    return true;
}

slot_file_result slot_disk_cache::persist(server_context & sc, server_slot & slot, const std::string & keep) {
    slot_file_result res;
    if (slot.cache_tokens.empty()) {
        return res;
    }
    const uint64_t h = tokens_hash(slot.cache_tokens);
    if (h == slot.disk_hash || h == failed_hash) {
        return res; // already on disk as it is, or there was no room for this same state
    }
    // how much it would cost to rebuild it from the files that are already there
    size_t covered = 0;
    for (const auto & e : entries) {
        covered = std::max(covered, reusable(e.tokens, e.ckpt_n_tokens, slot.cache_tokens));
    }
    if (slot.cache_tokens.size() < covered + DISK_MIN_SAVE) {
        return res;
    }

    // the previous saves of the same conversation are contained in this one: remove them (the base stays)
    for (size_t i = entries.size(); i-- > 0;) {
        const auto & e = entries[i];
        if (!e.base && e.path != keep && e.tokens.get_common_prefix_exact(slot.cache_tokens) == e.tokens.size()) {
            drop(i, "superseded by a more recent save of the same conversation");
        }
    }

    char name[32];
    snprintf(name, sizeof(name), "%016llx.slot", (unsigned long long) h);
    res = write_entry(sc, slot, (fs::path(dir) / name).string(), false, keep);
    if (!res.ok) {
        failed_hash = h;
        return res;
    }
    slot.disk_hash = h;
    return res;
}

// Writes the slot state to `path` and adds it to the index, making room first.
slot_file_result slot_disk_cache::write_entry(server_context & sc, server_slot & slot, const std::string & path, bool base,
                                              const std::string & keep) {
    slot_file_result res;
    // state + the checkpoints that really go into the file + metadata (tokens as JSON)
    size_t need = llama_state_seq_get_size(sc.ctx, slot.id, 0) + (1u << 20) + slot.cache_tokens.size() * 8;
    if (llama_context * ctx_mtp = slot_mtp_ctx(slot)) {
        need += llama_state_seq_get_size(ctx_mtp, slot.id, 0);
    }
    for (const auto * c : slot_file_ckpts(slot, base)) {
        need += c->data.size();
    }
    if (!make_room(need, keep)) {
        res.error = "not enough space";
        return res;
    }
    res = slot_file_save(sc.ctx, slot, path, model_fp, base);
    std::string err;
    slot_disk_entry e;
    if (res.ok && !read_entry(path, model_fp, e, err)) {
        res.ok = false;
        res.error = "the file just written cannot be read back: " + err;
    }
    if (!res.ok) {
        SRV_ERR("disk cache: saving %zu tokens failed: %s\n", slot.cache_tokens.size(), res.error.c_str());
        std::error_code ec;
        fs::remove(path, ec);
        return res;
    }
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].path == path) {
            entries.erase(entries.begin() + i);
            break;
        }
    }
    entries.push_back(std::move(e));
    SRV_INF("disk cache: saved %s of %zu tokens in %s, %.2f GiB in %.0f ms\n", base ? "the base (system prompt and tools)" : "the conversation",
            res.n_tokens, path.c_str(), res.n_bytes / 1073741824.0, res.ms);
    return res;
}

bool slot_disk_cache::restore(server_context & sc, server_slot & slot, const std::string & path) {
    size_t i = 0;
    while (i < entries.size() && entries[i].path != path) {
        ++i;
    }
    if (i == entries.size()) {
        return false;
    }
    const slot_file_result res = slot_file_load(sc.ctx, slot, path, model_fp);
    if (!res.ok) {
        SRV_ERR("disk cache: restoring %s failed, rereading from scratch: %s\n", path.c_str(), res.error.c_str());
        drop(i, "file rejected");
        slot.disk_hash = 0;
        return false;
    }
    std::error_code ec;
    fs::last_write_time(path, fs::file_time_type::clock::now(), ec);
    entries[i].t_used = now_s();
    slot.disk_hash = tokens_hash(slot.cache_tokens);
    SRV_INF("disk cache: restored %zu tokens from %s (%.2f GiB) in %.0f ms\n", res.n_tokens, path.c_str(),
            res.n_bytes / 1073741824.0, res.ms);
    return true;
}

void slot_disk_cache::prepare(server_context & sc, server_slot & slot, const server_task & task) {
    const server_tokens & prompt = task.tokens;
    slot.disk_base_pos = -1;
    if (prompt.empty()) {
        return;
    }

    const size_t own = reusable(slot.cache_tokens, slot_ckpt_n_tokens(slot), prompt);
    std::string best;
    size_t best_n = 0, best_bytes = 0;
    for (const auto & e : entries) {
        const size_t n = reusable(e.tokens, e.ckpt_n_tokens, prompt);
        if (n > best_n || (n == best_n && n > 0 && e.bytes < best_bytes)) {
            best = e.path;
            best_n = n;
            best_bytes = e.bytes;
        }
    }
    const bool swap = !best.empty() && best_n >= own + DISK_MIN_GAIN;
    // The state in the slot is about to be replaced, or overwritten before the end of the last prompt
    // (another conversation): it goes to disk first. The answer after the prompt does not matter: the next
    // turn of the same conversation rereads it anyway (without the reasoning).
    size_t valuable = slot.cache_tokens.size();
    for (const auto & c : slot.server_cached_prompt.checkpoints) {
        if (c.prompt_end) {
            valuable = std::min(valuable, (size_t) c.n_tokens);
        }
    }
    if (swap || valuable >= own + DISK_MIN_SAVE) {
        persist(sc, slot, best);
    }
    if (swap) {
        SLT_INF(slot, "disk cache: the file reuses %zu tokens of the prompt, the slot %zu\n", best_n, own);
        restore(sc, slot, best);
    }

    // the common prefix (system prompt + tools) is not on disk yet: it is saved when
    // the prompt read reaches it (see batch_pending_prompt)
    const std::string base_text = json_value(task.data, "cache_base_text", std::string());
    const size_t n_base = base_length(sc.ctx, base_text, prompt.tokens_data());
    if (n_base == 0) {
        return;
    }
    for (const auto & e : entries) {
        if (e.tokens.size() == n_base && e.tokens.get_common_prefix_exact(prompt) == n_base) {
            return; // it is already there
        }
    }
    slot.disk_base_pos = (int32_t) n_base;
}

void slot_disk_cache::save_base(server_context & sc, server_slot & slot) {
    const int32_t pos = slot.disk_base_pos;
    slot.disk_base_pos = -1;
    if ((int32_t) slot.cache_tokens.size() != pos) {
        // the cache is not the prompt from the start (context shift): the state is not the base
        SLT_WRN(slot, "disk cache: base not saved, %zu tokens in cache instead of %d\n", slot.cache_tokens.size(), pos);
        return;
    }
    write_entry(sc, slot, (fs::path(dir) / base_file_name(tokens_hash(slot.cache_tokens))).string(), true, "");
}

size_t slot_disk_cache::base_on_disk(llama_context * ctx, const std::string & base_text, const server_tokens & prompt) const {
    const llama_tokens ptok = prompt.tokens_data();
    const size_t n_base = base_length(ctx, base_text, ptok);
    if (n_base == 0) {
        return 0;
    }
    std::error_code ec;
    return fs::is_regular_file(fs::path(dir) / base_file_name(tokens_hash(ptok.data(), n_base)), ec) ? n_base : 0;
}

void slot_disk_cache::autosave(server_context & sc) {
    const int64_t now_us = ggml_time_us();
    for (auto & slot : sc.slots) {
        if (slot.available() && slot.t_last_used > 0 && now_us - slot.t_last_used >= DISK_IDLE_SAVE_S * 1000000) {
            persist(sc, slot);
        }
    }
}

void slot_disk_cache::restore_latest(server_context & sc, server_slot & slot) {
    const slot_disk_entry * latest = nullptr;
    for (const auto & e : entries) {
        if (!latest || e.t_used > latest->t_used) {
            latest = &e;
        }
    }
    if (latest) {
        const std::string path = latest->path; // restore() may remove the entry from the index
        restore(sc, slot, path);
    }
}
