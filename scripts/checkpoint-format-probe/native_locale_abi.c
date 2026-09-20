/* Independent SDK ABI oracle for the isolated Rust UCRT experiment. */
#include <locale.h>
#include <stddef.h>
#include <stdio.h>

int main(void) {
    printf("{\"size\":%zu,\"alignment\":%zu,\"grouping\":%zu,"
           "\"wide_decimal\":%zu,\"wide_separator\":%zu}\n",
           sizeof(struct lconv), (size_t)__alignof(struct lconv),
           offsetof(struct lconv, grouping), offsetof(struct lconv, _W_decimal_point),
           offsetof(struct lconv, _W_thousands_sep));
    return 0;
}
