#include "j128.h"

int j128_t::from_pylong(PyLongObject* obj)
{
    const int NB64 = 8;
    unsigned char bytes[2*NB64] = {};

#if PY_VERSION_HEX >= 0x030D0000
    int ret = PyLong_AsNativeBytes((PyObject*)obj, bytes, sizeof bytes, -1);
#else
    int ret = _PyLong_AsByteArray(obj, bytes, sizeof bytes, 1 /* little_endian */, 0 /* unsigned */);
#endif
    if (ret < 0)
	return ret;

    memcpy(&lo, bytes,        NB64);
    memcpy(&hi, bytes + NB64, NB64);
    return 0;
}


PyObject* j128_t::to_pylong() const
{
    const int NB64 = 8;
    unsigned char bytes[2*NB64];

    uint64_t x = lo;
    for (int i=0 ; i<NB64 ; i++) {
	bytes[i] = (unsigned char)(x & 0xff);
	x >>= 8;
    }
    x = hi;
    for (int i=0 ; i<NB64 ; i++) {
	bytes[i+NB64] = (unsigned char)(x & 0xff);
	x >>= 8;
    }

    return _PyLong_FromByteArray(bytes, sizeof bytes, true, false);
}


j128_t j128_t::mul10() const
{
    uint64_t lo_lo = lo & UINT32_MAX;
    uint64_t lo_hi = lo >> 32;
    uint64_t lo_lo_10 = 10 * lo_lo;
    uint64_t lo_hi_10 = 10 * lo_hi;

    j128_t out;
    out.hi = hi * 10 + (lo_hi_10 >> 32);
    out += lo_lo_10;
    out += (lo_hi_10 << 32);
    return out;
}


j128_t j128_t::div10(int& remainder) const
{
    const uint64_t T64_D_10 = UINT64_MAX / 10;
    const uint64_t T64_M_10 = 1 + (UINT64_MAX % 10);

    const uint64_t q_lo = lo / 10;
    const uint64_t r_lo = lo % 10;
    const uint64_t q_hi = hi / 10;
    const uint64_t r_hi = hi % 10;

    j128_t out;
    out.hi = q_hi;
    out.lo = T64_D_10 * r_hi;

    remainder = r_lo + r_hi * T64_M_10;
    out.lo +=  q_lo;
    out.lo += remainder / 10;
    remainder %= 10;
    return out;
}

const char* j128_t::to_string(char buf[DEC_STR_LEN]) const
{
    int i = DEC_STR_LEN-1;
    buf[i] = '\0';
    j128_t q = *this;
    int r;

    do {
	q = q.div10(r);
	i--;
	buf[i] = '0' + r;
    } while (q > 0);
    return buf + i;
}

j128_t j128_t::from_string(const char buf[DEC_STR_LEN])
{
    j128_t out = 0;
    const char* p = buf;
    while (*p >= '0' && *p <= '9') {
	out = out.mul10();
	out += (uint64_t)(*p - '0');
	p++;
    }
    return out;
}

#ifdef TEST

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main()
{
    j128_t one = 1;
    assert(one.hi == 0);
    assert(one.lo == 1);

    j128_t two = 2;
    assert(two.hi == 0);
    assert(two.lo == 2);

    j128_t three = one + two;
    assert(three.hi == 0);
    assert(three.lo == 3);

    j128_t thirty = three.mul10();
    assert(thirty.hi == 0);
    assert(thirty.lo == 30);

    int r;
    j128_t fifty_four = 54;
    j128_t five = fifty_four.div10(r);
    assert(five.hi == 0);
    assert(five.lo == 5);
    assert(r == 4);

    uint64_t big = 1;
    big <<= 63;
    j128_t as_big = big;
    j128_t roll = as_big + as_big;
    assert(roll.hi == 1);
    assert(roll.lo == 0);

    j128_t d10 = roll.div10(r);
    j128_t m10 = d10.mul10() + r;
    assert(m10 == roll);

    j128_t sub = roll - as_big;
    assert(sub == as_big);

    j128_t rp5 = roll + 5;
    j128_t rp1 = roll + 1;
    j128_t four = rp5 - rp1;
    assert(four == 4);

    char buf[j128_t::DEC_STR_LEN];
    assert(strcmp(fifty_four.to_string(buf), "54") == 0);

    const char* s = "91285735723312413301245362";
    j128_t fs = j128_t::from_string(s);
    assert(strcmp(fs.to_string(buf), s) == 0);

    printf("All Tests Passed\n");
    return 0;
}

#endif // TEST
