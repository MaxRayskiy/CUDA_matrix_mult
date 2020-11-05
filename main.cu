#include <iostream>

#include "headers/settings.h"
#include "headers/test.h"

int main() {
    std::cout << "BLOCKSIZE = " << BLOCKSIZE << std::endl;

    // RunTest(width, mid_size, height, print_time, is_debug)
    RunTest(64, 64, 64, true, false);
    RunTest(128, 64, 64, true, false);
    RunTest(64, 128, 64, true, false);
    RunTest(512, 64, 64, true, false);
    RunTest(256, 128, 128, true, false);
    RunTest(128, 512, 128, true, false);
    RunTest(512, 256, 256, true, false);
    RunTest(256, 1024, 256, true, false);

    RunTest(2048, 512, 512, true, false);
    RunTest(1024, 1024, 1024, true, false);
    RunTest(2048, 1024, 1024, true, false);

    RunTest(2048, 2048, 1024, true, false);
    RunTest(2048, 4096, 1024, true, false);
    RunTest(8192, 8192, 1024, true, false);

    return 0;
}
