#pragma once

// A slot's state in a file, and the disk cache of conversations (--cache-disk).
//
// A file contains, in this order:
//   1. the sequence state written by llama: tokens, KV of the attention layers, recurrent state
//      (DeltaNet) at the final position;
//   2. the useful checkpoints: the recurrent state at earlier points. The recurrent state cannot be
//      rewound, so a prompt that diverges before the end (the next turn renders the previous answer
//      without the reasoning) restarts from the end-of-prompt checkpoint instead of from zero;
//   3. with MTP on, the cache of the MTP layer (its context has its own cache): without it, the drafts
//      after a restore would see only the new tokens. Optional: a file saved without MTP also
//      loads with MTP (the MTP layer warms up on the new tokens) and vice versa;
//   4. the metadata (JSON): model hash, hash of each part, tokens with the images;
//   5. a 32-byte footer that locates the metadata.
// It is written to <file>.tmp and renamed at the end: a half-written file never has the real name.
// On restore every hash is checked again: a broken file, or one from another model, is rejected.

#include "server-context.h"

#include <cstdint>
#include <string>
#include <vector>

struct slot_file_result {
    bool        ok = false;
    std::string error;      // why it failed, for the log and for the HTTP response
    size_t      n_tokens = 0;
    size_t      n_bytes  = 0;
    double      ms       = 0.0;
};

// Hash of what makes a state reusable: the model file (size, first and last 4 MiB, which hold
// metadata and weights), the LoRA adapters, the cache types and the format version. A state with a
// different hash is not loaded.
uint64_t slot_file_model_fingerprint(const gpt_params & params);

// Writes the slot state. With `base` the state is the prefix common to the conversations (system
// prompt and tools) and the checkpoints are not needed.
slot_file_result slot_file_save(llama_context * ctx, const server_slot & slot, const std::string & path,
                                uint64_t model_fp, bool base);

// Loads the file into the slot. If it fails, the slot stays empty: never a half state.
slot_file_result slot_file_load(llama_context * ctx, server_slot & slot, const std::string & path, uint64_t model_fp);

// Disk cache: one file per state, identified by its tokens (not by who is talking). Before every
// request the prompt is compared with the state in the slot and with the files: if a file reuses
// more of it, the current state is saved (if it is not already) and the file is loaded. Side requests
// (title, summary, sub-agents) are conversations like the others. The common prefix (system prompt +
// tools) is saved as a "base" the first time it is read: a new conversation starts from there.
struct slot_disk_entry {
    std::string          path;
    server_tokens        tokens;
    std::vector<int64_t> ckpt_n_tokens; // where the file's checkpoints are
    bool                 base   = false;
    size_t               bytes  = 0;
    int64_t              t_used = 0;    // seconds, time of the last save or restore
};

struct slot_disk_cache {
    std::string dir;
    size_t      limit_bytes;
    int64_t     max_age_s;
    uint64_t    model_fp;
    std::vector<slot_disk_entry> entries;
    uint64_t    failed_hash = 0; // a state that did not fit: not retried until it changes

    slot_disk_cache(std::string dir, size_t limit_bytes, int64_t max_age_s, uint64_t model_fp);

    // Index the folder: drop leftover .tmp files and files that are expired, broken or from another model.
    void scan();

    // Before a completion request on the slot (see above).
    void prepare(server_context & sc, server_slot & slot, const server_task & task);

    // Save the slot if it changed and rereading it would be costly: `ok` false with empty `error` = nothing to do.
    // `keep` is not deleted even if the new state contains it.
    slot_file_result persist(server_context & sc, server_slot & slot, const std::string & keep = "");

    // With the slot idle for a while, save (so shutdowns and crashes do not lose the work).
    void autosave(server_context & sc);

    // At startup: load the most recently used state (the conversation open at shutdown).
    void restore_latest(server_context & sc, server_slot & slot);

    // The slot is at the end of the common prefix: save it as the base.
    void save_base(server_context & sc, server_slot & slot);

    // Is the base of this prompt already on disk? How many tokens it has, 0 if missing (or if the prompt has none).
    // Can be called from an HTTP thread: it only looks at the file, which depends on the base's tokens, and
    // does not touch the engine state.
    size_t base_on_disk(llama_context * ctx, const std::string & base_text, const server_tokens & prompt) const;

private:
    bool restore(server_context & sc, server_slot & slot, const std::string & path);
    slot_file_result write_entry(server_context & sc, server_slot & slot, const std::string & path, bool base,
                                 const std::string & keep);
    bool make_room(size_t need, const std::string & keep);
    void drop(size_t i, const char * why);
    void expire();
    size_t total_bytes() const;
};
