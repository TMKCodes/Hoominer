#pragma once

/* OpenCL kernel source embedded via xxd.
 *
 * The actual definitions live in `src/hoohash_cl_data.c` (which includes the
 * generated `src/hoohash_cl_data.inc`). This header is safe to include from
 * multiple translation units.
 */

extern unsigned char Hoohash_cl[];
extern unsigned int Hoohash_cl_len;
