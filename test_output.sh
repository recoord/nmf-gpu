#!/usr/bin/env bash

set -e

w_out_md5=$(md5sum "$1/Wout.bin" | cut -d ' ' -f 1)
w_test_md5=$(md5sum "$1/Wtest.bin" | cut -d ' ' -f 1)
h_out_md5=$(md5sum "$1/Hout.bin" | cut -d ' ' -f 1)
h_test_md5=$(md5sum "$1/Htest.bin" | cut -d ' ' -f 1)

if [ "$w_out_md5" != "$w_test_md5" ]; then
    echo "Wout.bin and Wtest.bin are different"
    exit 1
fi

if [ "$h_out_md5" != "$h_test_md5" ]; then
    echo "Hout.bin and Htest.bin are different"
    exit 1
fi

echo
echo "Result is identical to the test data"
