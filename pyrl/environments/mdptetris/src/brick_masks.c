/**
 * Definition of the bit masks declared in brick_masks.h
 */

#include "config.h"
#include "brick_masks.h"

const uint16_t brick_masks[] = {
  0x8000, /* X............... */
  0x4000, /* .X.............. */
  0x2000, /* ..X............. */
  0x1000, /* etc */
  0x0800,
  0x0400,
  0x0200,
  0x0100,
  0x0080,
  0x0040,
  0x0020,
  0x0010,
  0x0008,
  0x0004,
  0x0002, /* ..............X. */
  0x0001  /* ...............X */
};

const uint16_t brick_masks_inv[] = {
   0x7FFF, /* .XXXXXXXXXXXXXXX */
  ~0x4000, /* X.XXXXXXXXXXXXXX */
  ~0x2000, /* XX.XXXXXXXXXXXXX */
  ~0x1000, /* etc */
  ~0x0800,
  ~0x0400,
  ~0x0200,
  ~0x0100,
  ~0x0080,
  ~0x0040,
  ~0x0020,
  ~0x0010,
  ~0x0008,
  ~0x0004,
  ~0x0002, /* XXXXXXXXXXXXXX.X */
  ~0x0001  /* XXXXXXXXXXXXXXX. */
};

/**
 * Prints the 16 bits of a row into a file.
 * @param out the file to write
 * @param row the row
 */
void print_row(FILE *out, uint16_t row) {
  int i;
  for (i = 0; i < 16; i++) {
    if (row & brick_masks[i]) {
      fprintf(out, "X");
    }
    else {
      fprintf(out, ".");
    }
  }
}
