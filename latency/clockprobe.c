// Runs on the device (adb shell): for each byte read on stdin, print the
// device CLOCK_MONOTONIC in nanoseconds. measure.py times these round trips to
// map device dates to computer dates.
#include <stdio.h>
#include <time.h>
#include <unistd.h>

int
main(void) {
    char c;
    while (read(0, &c, 1) == 1) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        char line[32];
        int n = snprintf(line, sizeof(line), "%lld\n",
                         (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec);
        if (write(1, line, n) != n) {
            return 1;
        }
    }
    return 0;
}
