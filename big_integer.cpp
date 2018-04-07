#include "big_integer.h"

#include <cstring>
#include <stdexcept>

#include <algorithm>

const unsigned int MAX = UINT32_MAX;
const unsigned int BASE = 32;
const int BASE10 = 1000000000;

using std::vector;

template<typename T>
unsigned int castUnsignedInt(T x) {
    return static_cast<unsigned int>(x &
                                     MAX); // оставляет у числа младшие 32 бита
}

template<typename T>
unsigned long long castUnsignedLongLong(T x) {
    return static_cast<unsigned long long>(x);
}

void big_integer::make_fit() {
    while (!data.empty() && ((sign && data.back() == MAX) || (!sign && data.back() == 0))) {
        data.pop_back();
    }
}

big_integer::big_integer(bool new_sign, vector<unsigned int> const &new_data) : sign(new_sign), data(new_data) {
    make_fit();
}

void big_integer::swap(big_integer &other) noexcept {
    std::swap(data, other.data);
    std::swap(sign, other.sign);
}

bool big_integer::zero() const {
    return (!sign) && (length() == 0);
}

size_t big_integer::length() const {
    return data.size();
}

big_integer big_integer::abs() const {
    return sign ? -(*this) : *this;
}

big_integer::big_integer() : sign(false) {}

big_integer::big_integer(big_integer const &other) : sign(other.sign), data(other.data) {
    make_fit();
    //mpz_init_set(mpz, other.mpz);
}

big_integer::big_integer(int a) : sign(a < 0), data(1) {
    data[0] = castUnsignedInt(a);
    make_fit();
    //mpz_init_set_si(mpz, a);
}

big_integer::big_integer(unsigned int a) : sign(0), data(1) {
    data[0] = a;
    make_fit();
}

big_integer::big_integer(unsigned long long a) : sign(0), data(2) {
    data[0] = castUnsignedInt(a);
    data[1] = castUnsignedInt(a >> BASE);   // храним, начиная с младших разрядов
    make_fit();
}

unsigned int big_integer::digit(size_t i) const {   // достает с проверкой на выход за пределы вектора
    if (i < length()) {
        return data[i];
    } else if (sign) {
        return MAX;
    } else {
        return 0;
    }
}

unsigned int big_integer::digitReal(size_t i) const {   // достает беззнаковое 32битное беззнаковое число (типа цифру) из вектора
    return data[i];
}

/*
big_integer::big_integer(std::string const &str) {
    if (mpz_init_set_str(mpz, str.c_str(), 10)) {
        mpz_clear(mpz);
        throw std::runtime_error("invalid string");
    }
}

big_integer::~big_integer() {
    mpz_clear(mpz);
}*/

big_integer &big_integer::operator=(big_integer const &other) {
    big_integer temp(other);
    swap(temp);
    //mpz_set(mpz, other.mpz);
    return *this;
}

big_integer &big_integer::operator+=(big_integer const &rhs) {
    return *this = *this + rhs;
    //mpz_add(mpz, mpz, rhs.mpz);
    //return *this;
}

big_integer &big_integer::operator-=(big_integer const &rhs) {
    return *this = *this - rhs;
    //mpz_sub(mpz, mpz, rhs.mpz);
    //return *this;
}

big_integer &big_integer::operator*=(big_integer const &rhs) {
    return *this = *this * rhs;
    //mpz_mul(mpz, mpz, rhs.mpz);
    //return *this;
}

big_integer &big_integer::operator/=(big_integer const &rhs) {
    return *this = *this / rhs;
    //mpz_tdiv_q(mpz, mpz, rhs.mpz);
    //return *this;
}

big_integer &big_integer::operator%=(big_integer const &rhs) {
    return *this = *this % rhs;
    //mpz_tdiv_r(mpz, mpz, rhs.mpz);
    //return *this;
}

big_integer &big_integer::operator&=(big_integer const &rhs) {
    return *this = *this & rhs;
    //mpz_and(mpz, mpz, rhs.mpz);
    //return *this;
}

big_integer &big_integer::operator|=(big_integer const &rhs) {
    return *this = *this | rhs;
    //mpz_ior(mpz, mpz, rhs.mpz);
    //return *this;
}

big_integer &big_integer::operator^=(big_integer const &rhs) {
    return *this = *this ^ rhs;
    //mpz_xor(mpz, mpz, rhs.mpz);
    //return *this;
}

big_integer &big_integer::operator<<=(unsigned int rhs) {
    return *this = *this << rhs;
    //mpz_mul_2exp(mpz, mpz, rhs);
    //return *this;
}

big_integer &big_integer::operator>>=(unsigned int rhs) {
    return *this = *this >> rhs;
    //mpz_div_2exp(mpz, mpz, rhs);
    //return *this;
}

big_integer big_integer::operator+() const {
    return *this;
}

big_integer big_integer::operator-() const {
    if (zero()) {
        return *this;
    } else if (length() == 0) {
        return big_integer(1u);
    }
    size_t n = length() + 2;
    vector<unsigned int> temp(n);
    unsigned long long sum = castUnsignedInt(~digit(0)) + 1ULL; // именно так в нашем представлении берется signed унарный минус для представления в unsigned
    unsigned long long carry = sum >> BASE;  //остаточек
    temp[0] = castUnsignedInt(sum);
    for (size_t i = 1; i < n - 2; ++i) {
        sum = carry + castUnsignedLongLong(~digitReal(i));
        temp[i] = castUnsignedInt(sum);
        carry = sum >> BASE;
    }
    for (size_t i = n - 2; i < n; ++i) {
        sum = (carry + (sign ? 0 : MAX));
        temp[i] = castUnsignedInt(sum);
        carry = sum >> BASE;
    }
    return big_integer(temp.back() & (1 << (BASE - 1)/*проверяем бит под знак у temp.back()*/), temp);
    //big_integer r;
    //mpz_neg(r.mpz, mpz);
    //return r;
}

big_integer big_integer::operator~() const {
    vector<unsigned int> temp(data);
    for (size_t i = 0; i < data.size(); ++i) {
        temp[i] = ~data[i];
    }
    return big_integer(!sign, temp);
    //big_integer r;
    //mpz_com(r.mpz, mpz);
    //return r;
}

big_integer &big_integer::operator++() {
    *this += 1;
    return *this;
    //mpz_add_ui(mpz, mpz, 1);
    //return *this;
}

big_integer big_integer::operator++(int) {
    big_integer r = *this;
    ++*this;
    return r;
}

big_integer &big_integer::operator--() {
    *this -= 1;
    return *this;
    //mpz_sub_ui(mpz, mpz, 1);
    //return *this;
}

big_integer big_integer::operator--(int) {
    big_integer r(*this);
    --(*this);
    return r;
}

big_integer operator&(big_integer const &a, big_integer const &b) {
    size_t maxLength = std::max(a.length(), b.length());
    size_t minLength = std::min(a.length(), b.length());
    vector<unsigned int> temp(maxLength);
    for (size_t i = 0; i < minLength; i++) {
        temp[i] = a.digitReal(i) & b.digitReal(i);
    }
    for (size_t i = minLength; i < maxLength; i++) {
        temp[i] = a.digit(i) & b.digit(i);
    }
    return big_integer(a.sign & b.sign, temp);
}

big_integer operator|(big_integer const &a, big_integer const &b) {
    size_t maxLength = std::max(a.length(), b.length());
    size_t minLength = std::min(a.length(), b.length());
    vector<unsigned int> temp(maxLength);
    for (size_t i = 0; i < minLength; i++) {
        temp[i] = a.digitReal(i) | b.digitReal(i);
    }
    for (size_t i = minLength; i < maxLength; i++) {
        temp[i] = a.digit(i) | b.digit(i);
    }
    return big_integer(a.sign | b.sign, temp);
}

big_integer operator^(big_integer const &a, big_integer const &b) {
    size_t maxLength = std::max(a.length(), b.length());
    size_t minLength = std::min(a.length(), b.length());
    vector<unsigned int> temp(maxLength);
    for (size_t i = 0; i < minLength; i++) {
        temp[i] = a.digitReal(i) ^ b.digitReal(i);
    }
    for (size_t i = minLength; i < maxLength; i++) {
        temp[i] = a.digit(i) ^ b.digit(i);
    }
    return big_integer(a.sign ^ b.sign, temp);
}

big_integer operator<<(big_integer const &a, unsigned int b) {
    if (b == 0) {
        return big_integer(a);
    }
    size_t div = b >> 5;    // если b>32, то div не равен 0 : сдвиг на дофига (больше, чем на одну целую ячейку), div == количество этих целых
    size_t mod = b % BASE;  // size_t mod = b & (BASE - 1); is another way to find mod
    size_t new_size = a.length() + div + 1; // + 1, т.к. mod != 0 обычно
    vector<unsigned int> temp(new_size);
    temp[div] = castUnsignedInt((unsigned long long) (a.digit(0)) << mod);
    for (size_t i = div + 1; i < new_size; i++) {
        unsigned long long x = (unsigned long long) (a.digit(i - div)) << mod;
        unsigned long long y = (unsigned long long) (a.digitReal(i - div - 1)) >> (BASE - mod);
        temp[i] = castUnsignedInt(x | y);
    }
    return big_integer(a.sign, temp);
}

big_integer operator>>(big_integer const &a, unsigned int b) {
    if (b == 0) {
        return big_integer(a);
    }
    size_t div = b >> 5;
    size_t mod = b % BASE;  // size_t mod = b & (BASE - 1); is another way to find mod
    size_t new_size = 0;
    if (div < a.length()) {
        new_size = a.length() - div;
    }
    vector<unsigned int> temp(new_size);
    for (size_t i = 0; i < new_size; i++) {
        unsigned long long x = (unsigned long long) (a.digitReal(i + div)) >> mod;
        unsigned long long y = (unsigned long long) (a.digit(i + div + 1)) << (BASE - mod);
        temp[i] = castUnsignedInt(x | y);
    }
    return big_integer(a.sign, temp);
}

/*
big_integer operator+(big_integer a, big_integer const &b) {
    return a += b;
}

big_integer operator-(big_integer a, big_integer const &b) {
    return a -= b;
}

big_integer operator*(big_integer a, big_integer const &b) {
    return a *= b;
}

big_integer operator/(big_integer a, big_integer const &b) {
    return a /= b;
}

big_integer operator%(big_integer a, big_integer const &b) {
    return a %= b;
}

big_integer operator&(big_integer a, big_integer const &b) {
    return a &= b;
}

big_integer operator|(big_integer a, big_integer const &b) {
    return a |= b;
}

big_integer operator^(big_integer a, big_integer const &b) {
    return a ^= b;
}

big_integer operator<<(big_integer a, int b) {
    return a <<= b;
}

big_integer operator>>(big_integer a, int b) {
    return a >>= b;
}*/

bool operator==(big_integer const &a, big_integer const &b) {
    return (a.sign == b.sign) && (a.data == b.data);
    //return mpz_cmp(a.mpz, b.mpz) == 0;
}

bool operator!=(big_integer const &a, big_integer const &b) {
    return !(a == b);
    //return mpz_cmp(a.mpz, b.mpz) != 0;
}

bool operator<(big_integer const &a, big_integer const &b) {
    if (a.sign != b.sign) {
        return a.sign;
    }
    if (a.length() != b.length()) {
        return a.length() < b.length();
    }
    //size_t temp = a.length(); // does not improve the work of the cycle - c++ optimisation make the same
    for (size_t i = a.length(); i > 0; i--) {   // костыли и велосипеды, программируем как умеем. size_t для мужиков
        if (a.digit(i - 1) != b.digit(i - 1)) {
            return a.digitReal(i - 1) < b.digitReal(i - 1);
        }
    }
    return 0;
    //return mpz_cmp(a.mpz, b.mpz) < 0;
}

bool operator>(big_integer const &a, big_integer const &b) {
    return b < a;
    //return mpz_cmp(a.mpz, b.mpz) > 0;
}

bool operator<=(big_integer const &a, big_integer const &b) {
    return !(a > b);
    //return mpz_cmp(a.mpz, b.mpz) <= 0;
}

bool operator>=(big_integer const &a, big_integer const &b) {
    return !(a < b);
    //return mpz_cmp(a.mpz, b.mpz) >= 0;
}

std::string to_string(big_integer const &a) {
    if (a.zero()) {
        return "0";
    } else if (a.length() == 0) {
        return "-1";
    }
    std::string res = "";
    big_integer abs_a(a.abs());
    while (!abs_a.zero()) {
        unsigned int temp = (abs_a % BASE10).digit(0);  // BASE10 потому что unsigned int тип у temp
        for (size_t i = 0; i < 9; i++) {
            res.push_back('0' + temp % 10); // кладет символ-цифру с правильным номером
            temp /= 10;
        }
        abs_a /= BASE10;
    }
    while (!res.empty() && res.back() == '0') {     // уберём лишние нули справа
        res.pop_back();
    }
    if (a.sign) {
        res.push_back('-');
    }
    reverse(res.begin(), res.end());    // хранили же в data с младшего разряда, записали обратное число, перевернем его!
    return res;
    /*char *tmp = mpz_get_str(NULL, 10, a.mpz);
    std::string res = tmp;

    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);

    freefunc(tmp, strlen(tmp) + 1);

    return res;*/
}
/*
std::ostream &operator<<(std::ostream &s, big_integer const &a) {
    return s << to_string(a);
}*/

big_integer operator+(big_integer const &a, big_integer const &b) {
    size_t maxLength = std::max(a.length(), b.length()) + 2;
    size_t minLength = std::min(a.length(), b.length());
    vector<unsigned int> temp(maxLength);
    unsigned long long carry = 0;
    unsigned long long sum = 0;

    for (size_t i = 0; i < minLength; ++i) {
        sum = (carry + a.digitReal(i)) + b.digitReal(i);
        temp[i] = castUnsignedInt(sum);
        carry = sum >> BASE;
    }
    for (size_t i = minLength; i < maxLength; ++i) {
        sum = (carry + a.digit(i)) + b.digit(i);
        temp[i] = castUnsignedInt(sum);
        carry = sum >> BASE;
    }
    return big_integer(temp.back() & (1 << (BASE - 1)), temp);
}

big_integer operator-(big_integer const &a, big_integer const &b) {
    size_t maxLength = std::max(a.length(), b.length()) + 3;
    size_t minLength = std::min(a.length(), b.length());
    vector<unsigned int> temp(maxLength);
    unsigned long long carry = 0;
    unsigned long long sum = 0;

    if (minLength > 0) {
        sum = castUnsignedLongLong(a.digitReal(0)) + 1ULL + castUnsignedLongLong(~b.digitReal(0));
        temp[0] = castUnsignedInt(sum);
        carry = sum >> BASE;
        for (size_t i = 1; i < minLength; ++i) {
            sum = carry + castUnsignedLongLong(a.digitReal(i)) + castUnsignedLongLong(~b.digitReal(i));
            temp[i] = castUnsignedInt(sum);
            carry = sum >> BASE;
        }
        for (size_t i = minLength; i < maxLength; ++i) {
            sum = carry + castUnsignedLongLong(a.digit(i)) + castUnsignedLongLong(~b.digit(i));
            temp[i] = castUnsignedInt(sum);
            carry = sum >> BASE;
        }
    } else {
        sum = castUnsignedLongLong(a.digit(0)) + 1ULL + castUnsignedLongLong(~b.digit(0));
        temp[0] = castUnsignedInt(sum);
        carry = sum >> BASE;
        for (size_t i = 1; i < maxLength; ++i) {
            sum = carry + castUnsignedLongLong(a.digit(i)) + castUnsignedLongLong(~b.digit(i));
            temp[i] = castUnsignedInt(sum);
            carry = sum >> BASE;
        }
    }

    return big_integer(temp.back() & (1 << (BASE - 1)), temp);
}

void mul_vector(vector<unsigned int> &res, vector<unsigned int> const &a, vector<unsigned int> const &b) {
    size_t aLength = a.size();
    size_t bLength = b.size();
    for (size_t i = 0; i < aLength; i++) {
        unsigned long long carry = 0, mul = 0, tmp = 0;
        for (size_t j = 0; j < bLength; j++) {
            size_t k = i + j;
            mul = (unsigned long long) (a[i]) * b[j];
            tmp = (mul & MAX) + res[k] + carry;
            res[k] = castUnsignedInt(tmp);
            carry = (mul >> BASE) + (tmp >> BASE);
        }
        res[i + bLength] += castUnsignedInt(carry);
    }
}

void mul_big_small(vector<unsigned int> &res, vector<unsigned int> const &a, const unsigned int b) {
    size_t aLength = a.size();
    res.resize(aLength + 1);
    unsigned long long carry = 0, mul = 0, tmp = 0;
    for (size_t i = 0; i < aLength; i++) {
        mul = (unsigned long long) (a[i]) * b;
        tmp = (mul & MAX) + carry;
        res[i] = castUnsignedInt(tmp);
        carry = (mul >> BASE) + (tmp >> BASE);
    }
    res[aLength] = castUnsignedInt(carry);
}

void big_integer::correct() {
    if (!sign) {
        return;
    } else if (length() == 0) {
        sign = !sign;
        return;
    }
    size_t n = length();
    unsigned long long sum = castUnsignedLongLong(~data[0]) + 1ULL, carry = sum >> BASE;
    data[0] = castUnsignedInt(sum);
    for (size_t i = 1; i < n; i++) {
        sum = carry + castUnsignedLongLong(~data[i]);
        data[i] = castUnsignedInt(sum);
        carry = sum >> BASE;
    }
    data.push_back(castUnsignedInt(carry + MAX));
    make_fit();
}


big_integer operator*(big_integer const &a, big_integer const &b) {
    if (a.zero() || b.zero()) {
        return big_integer(0u);
    }
    big_integer abs_a(a.abs());
    big_integer abs_b(b.abs());
    if (abs_a.length() > abs_b.length()) {
        abs_a.swap(abs_b);
    }
    size_t aLength = abs_a.length();
    size_t bLength = abs_b.length();
    size_t len = (aLength + bLength + 1);
    vector<unsigned int> temp(len);
    if (abs_a.length() == 1) {
        mul_big_small(temp, abs_b.data, abs_a.digitReal(0));
    } else {
        mul_vector(temp, abs_a.data, abs_b.data);
    }
    big_integer res(a.sign ^ b.sign, temp);
    res.correct();
    return res;
}

int string_to_int(std::string const &s) {
    int ans = 0;
    for (auto a : s) {
        if (a < '0' || a > '9') {
            throw std::runtime_error("Char is incorrect");
        }
        ans = ans * 10 + (a - '0');
    }
    return ans;
}

int dec_pow(unsigned int st) {
    if (st == 0) {
        return 1;
    }
    if (st & 1) {
        return dec_pow(st - 1) * 10;
    }
    int preAns = dec_pow(st >> 1);
    return preAns * preAns;
}

big_integer string_to_number(std::string const &s) {
    big_integer res(0);
    size_t j = 0;
    if (s[j] == '-') {
        j++;
    }
    for (size_t i = j; i + 9 <= s.length(); i += 9) {
        res = res * BASE10 + string_to_int(s.substr(i, 9));
    }
    unsigned int mod = (s.length() - j) % 9;
    if (mod > 0) {
        res = res * dec_pow(mod) + string_to_int(s.substr(s.length() - mod, mod));
    }
    return j > 0 ? -res : res;
}

big_integer::big_integer(std::string const &str) : big_integer(string_to_number(str)) {}

unsigned int get_two(const unsigned int a, const unsigned int b, const unsigned int c) {
    unsigned long long res = a;
    res = ((res << BASE) + b) / c;
    if (res > MAX) {
        res = MAX;
    }
    return castUnsignedInt(res);
}

void sub_equal_vectors(vector<unsigned int> &a, vector<unsigned int> const &b) {
    unsigned long long sum = castUnsignedLongLong(~b[0]) + castUnsignedLongLong(a[0]) + 1LL, carry = sum >> BASE;
    a[0] = castUnsignedInt(sum);
    for (size_t i = 1; i < b.size(); i++) {
        sum = castUnsignedLongLong(~b[i]) + castUnsignedLongLong(a[i]) + carry;
        a[i] = castUnsignedInt(sum);
        carry = sum >> BASE;
    }
}

bool compare_equal_vectors(vector<unsigned int> const &a, vector<unsigned int> const &b) {
    for (size_t i = a.size(); i > 0; i--) {
        if (a[i - 1] != b[i - 1]) {
            return a[i - 1] < b[i - 1];
        }
    }
    return 0;
}

big_integer operator/(big_integer const &a, big_integer const &b) {
    if (b.zero()) {
        throw std::runtime_error("Division by zero");
    }
    big_integer abs_a(a.abs());
    big_integer abs_b(b.abs());
    if (abs_a < abs_b) {
        return 0;
    }

    // abs_a >= abs_b
    const unsigned int f = castUnsignedInt(
            ((unsigned long long) (MAX) + 1) / ((unsigned long long) (abs_b.data.back()) + 1));
    const size_t aLength = abs_a.length();
    const size_t bLength = abs_b.length();
    mul_big_small(abs_a.data, abs_a.data, f);
    mul_big_small(abs_b.data, abs_b.data, f);
    abs_a.make_fit();
    abs_b.make_fit();

    const size_t len = aLength - bLength + 1;   // длина результата деления
    const unsigned int divisor = abs_b.data.back();
    vector<unsigned int> temp(len);
    vector<unsigned int> dev(bLength + 1), div(bLength + 1, 0/*заполнить нулями*/);
    for (size_t i = 0; i < bLength; i++) {
        dev[i] = abs_a.digitReal(aLength + i - bLength);    // correct because abs_a >= abs_b and aLength >= bLength
    }
    dev[bLength] = abs_a.digit(aLength);    // dev хранит последние bLength+1 ячеек вектора abs_a
    if (bLength == 1) {
        for (size_t i = 0; i < len; i++) {
            dev[0] = abs_a.digitReal(aLength - bLength - i);    // держит i-ую с конца цифру (число 32битное) abs_a
            size_t resPos = len - 1 - i;
            unsigned int smallRes = get_two(dev[bLength], dev[bLength - 1], divisor); // деление в столбик так работает, две цифры (числа 32битных) - с запасом (иногда как раз)
            mul_big_small(div, abs_b.data, smallRes);   // результат умножения вектора abs_b.data на smallRes в div. В div делимое - остаток от строчки выше
            sub_equal_vectors(dev, div);    // остаток в dev
            for (size_t j = bLength; j > 0; j--) {  // сдвигаем dev
                dev[j] = dev[j - 1];
            }
            temp[resPos] = smallRes;
        }
    } else {    // если делитель больше одного 32битного числа
        for (size_t i = 0; i < len; i++) {
            dev[0] = abs_a.digitReal(aLength - bLength - i);
            size_t resPos = len - 1 - i;
            unsigned int smallRes = get_two(dev[bLength], dev[bLength - 1], divisor); // деление в столбик так работает, два куска по длине делителя - с запасом (иногда как раз)
            mul_big_small(div, abs_b.data, smallRes);   // результат умножения вектора abs_b.data на smallRes в div. В div делимое - остаток от строчки выше
            while ((smallRes >= 0) && compare_equal_vectors(dev, div) /*и не равны*/) {
                mul_big_small(div, abs_b.data, --smallRes);
            }
            sub_equal_vectors(dev, div);    // остаток в dev
            for (size_t j = bLength; j > 0; j--) {  //сдвигаем dev
                dev[j] = dev[j - 1];
            }
            temp[resPos] = smallRes;
        }
    }
    big_integer res(a.sign ^ b.sign, temp);
    res.correct();
    return res;
}

big_integer operator%(big_integer const &a, big_integer const &b) {
    return a - (a / b) * b;
}
