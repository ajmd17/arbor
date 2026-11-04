/*
 * arbor.h: the C API over arbor-core (crates/arbor-ffi/src/lib.rs).
 *
 * A recipe is one preset with its seed: a tree, shrub or vine species, a clump of
 * ground cover, or a rock. Its controls are the viewer's panel as a flat list, and
 * generating it gives a .glb with its textures inside and, for trees and cover, the
 * ARBOR_tree_wind sway data.
 *
 * A recipe may be used from any thread, one at a time. Clone it to grow it on another
 * thread; clones share the textures already prepared. Calls that fail return false or
 * NULL and leave the reason in arbor_last_error().
 *
 * Link with the libraries listed in native-libs.txt beside the library.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    ARBOR_TREE = 0,
    ARBOR_COVER = 1,
    ARBOR_ROCK = 2,
};

enum
{
    ARBOR_CONTROL_RANGED = 0,
    ARBOR_CONTROL_COUNT = 1,
    ARBOR_CONTROL_CHOICE = 2,
    ARBOR_CONTROL_TOGGLE = 3,
};

typedef struct ArborRecipe ArborRecipe;

typedef struct ArborBytes
{
    uint8_t* data;
    size_t len;
} ArborBytes;

/* The strings belong to the recipe and last until its controls change shape (see
 * arbor_control_set) or it is freed. */
typedef struct ArborControlInfo
{
    const char* id;
    const char* section;
    const char* group;
    const char* label;
    const char* help;
    uint32_t kind;
    float min; /* ranged: the slider's span */
    float max;
    uint32_t log;
    uint32_t count_min; /* count: the slider's span */
    uint32_t count_max;
    uint32_t option_count; /* choice: read each with arbor_control_option */
    uint32_t regrows;      /* zero for what only the shaders read */
} ArborControlInfo;

/* Ranged controls use lo, hi and is_range; counts, choices and toggles use integer.
 * landed is read-only: where this seed lands the range, the value to keep when a range
 * is folded back to one number. */
typedef struct ArborValue
{
    uint32_t is_range;
    float lo;
    float hi;
    float landed;
    uint32_t integer;
} ArborValue;

const char* arbor_last_error(void);

/* Where tree textures are read from; NULL forgets it. Cover and rocks bake their own. */
bool arbor_set_texture_dir(const char* dir);

uint32_t arbor_preset_count(uint32_t kind);
const char* arbor_preset_name(uint32_t kind, uint32_t i);

ArborRecipe* arbor_recipe_from_preset(uint32_t kind, const char* name);
ArborRecipe* arbor_recipe_from_ron(uint32_t kind, const char* ron);
ArborRecipe* arbor_recipe_clone(ArborRecipe* recipe);
void arbor_recipe_free(ArborRecipe* recipe);
uint32_t arbor_recipe_kind(ArborRecipe* recipe);
/* Both last until the next call returning text from the same recipe. */
const char* arbor_recipe_name(ArborRecipe* recipe);
const char* arbor_recipe_to_ron(ArborRecipe* recipe);

uint64_t arbor_get_seed(ArborRecipe* recipe);
bool arbor_set_seed(ArborRecipe* recipe, uint64_t seed);

uint32_t arbor_control_count(ArborRecipe* recipe);
bool arbor_control_info(ArborRecipe* recipe, uint32_t i, ArborControlInfo* out);
const char* arbor_control_option(ArborRecipe* recipe, uint32_t i, uint32_t option);
bool arbor_control_get(ArborRecipe* recipe, uint32_t i, ArborValue* out);
/* reshaped (may be NULL) says whether the control list changed shape and has to be
 * read again; the old list's strings are then gone. */
bool arbor_control_set(ArborRecipe* recipe, uint32_t i, const ArborValue* value, bool* reshaped);

bool arbor_generate_glb(ArborRecipe* recipe, ArborBytes* out);
void arbor_bytes_free(ArborBytes* bytes);

#ifdef __cplusplus
}
#endif
