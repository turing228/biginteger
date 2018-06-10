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

void big_integer::make_fit() {  // убирает лишние 0 из начала
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

unsigned int
big_integer::digitReal(size_t i) const {   // достает беззнаковое 32битное беззнаковое число (типа цифру) из вектора
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
    unsigned long long sum = castUnsignedInt(~digit(0)) +
                             1ULL; // именно так в нашем представлении берется signed унарный минус для представления в unsigned
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
    size_t div = b
            >> 5;    // если b>32, то div не равен 0 : сдвиг на дофига (больше, чем на одну целую ячейку), div == количество этих целых
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
    reverse(res.begin(),
            res.end());    // хранили же в data с младшего разряда, записали обратное число, перевернем его!
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

/*
unsigned int get_three(const unsigned int a1, const unsigned int a2, const unsigned int a3, const unsigned int b1, const unsigned int b2) {
    unsigned int preAns;
    unsigned int f = BASE10 / ();
    return castUnsignedInt(res);
}*/


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

vector<unsigned int> div_big_small(vector<unsigned int> const &a, const unsigned int b) {
    if (b == 0) {
        throw std::runtime_error("Division by zero");
    }
    size_t aLength = a.size();
    vector<unsigned int> res;
    res.assign(aLength, 0);
    //res.resize(aLength);
    unsigned long long int carry = 0;
    for (int i = aLength - 1; i >= 0; i--) {
        unsigned long long int temp = carry * (((unsigned long long int) 1) << 32) + a[i];
        res[i] = temp / b;
        carry = temp % b;
    }
    return res;
}

/*
unsigned int trial(vector<unsigned int> a, vector<unsigned int> b){
    const size_t aLength = a.size();
    const size_t bLength = b.size();
    if (2<=aLength) {
        size_t km = aLength+bLength;
        unsigned int r3 = (a[km]*BASE10+a[km-1])*BASE10+a[km-2];
        unsigned int d2 = b[]
    }
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
    //const unsigned int f = castUnsignedInt(
    //        ((unsigned long long) (MAX) + 1) / ((unsigned long long) (abs_b.data.back()) + 1));
    const size_t aLength = abs_a.length();
    const size_t bLength = abs_b.length();

    unsigned int f = BASE10/(b.digit(bLength-1)+1);
    big_integer r, d;
    mul_big_small(r.data, a.data, f);
    mul_big_small(d.data, b.data, f);
    //unsigned int q=0;
    big_integer dq, q;
    unsigned int qt=0;
    for (int i=aLength-bLength; i>=0; i--){
        if ((2<=bLength)&&(i+bLength<=aLength)){
            qt = trial(r.data, d.data);
            mul_big_small(dq.data, d.data, qt);
            if (r<dq){
                qt = qt-1;
                mul_big_small(dq.data, d.data, qt);
            }
            q.data[r.length()-1]=qt;
            sub_equal_vectors(r.data, dq.data);
        }
    }
    div_big_small(r.data, r.data, f);

    //17 page
}*/

big_integer div_nice(big_integer const &a, big_integer const &b) {
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
    vector<unsigned int> dev(bLength + 1), div(bLength + 1, 0);
    for (size_t i = 0; i < bLength; i++) {
        dev[i] = abs_a.digitReal(aLength + i - bLength);    // correct because abs_a >= abs_b and aLength >= bLength
    }
    dev[bLength] = abs_a.digit(aLength);    // dev хранит последние bLength+1 ячеек вектора abs_a
    if (bLength == 1) {
        for (size_t i = 0; i < len; i++) {
            dev[0] = abs_a.digitReal(aLength - bLength - i);    // держит i-ую с конца цифру (число 32битное) abs_a
            size_t resPos = len - 1 - i;
            unsigned int smallRes = get_two(dev[bLength], dev[bLength - 1],
                                            divisor); // деление в столбик так работает, две цифры (числа 32битных) - с запасом (иногда как раз)
            mul_big_small(div, abs_b.data,
                          smallRes);   // результат умножения вектора abs_b.data на smallRes в div. В div делимое - остаток от строчки выше
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
            unsigned int smallRes = get_two(dev[bLength], dev[bLength - 1],
                                            divisor); // деление в столбик так работает, два куска по длине делителя - с запасом (иногда как раз)
            mul_big_small(div, abs_b.data,
                          smallRes);   // результат умножения вектора abs_b.data на smallRes в div. В div делимое - остаток от строчки выше
            while ((smallRes >= 0) && compare_equal_vectors(dev, div)) {
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

void big_integer::my_push(unsigned int t) {
    data.push_back(t);
}

big_integer find_divider_pro(big_integer &d, size_t m) {
    big_integer d2;
    if (m >= 2) {
        d2.my_push(d.digitReal(m - 2));
    }
    d2.my_push(d.digitReal(m - 1));

    big_integer nice;
    nice.my_push(0);
    nice.my_push(0);
    nice.my_push(0);
    nice.my_push(1);
    return div_nice(nice, d2);
}

unsigned int trial(big_integer &r, big_integer &divider, size_t k, size_t m) {
    if (2 <= m && m <= k + m) {
        size_t km = k + m;

        /*big_integer _r3;
        big_integer r_km_1 = r.digitReal(km-1);
        mul_big_small(_r3.data, r_km_1.data, MAX);
        _r3 = _r3 + big_integer(r.digitReal(km - 2));
        big_integer __r3;
        mul_big_small(__r3.data, _r3.data, MAX);
        __r3 = _r3 + big_integer(r.digitReal(km - 3));
        big_integer r3 = __r3;*/

        big_integer r3;

        size_t rL = r.length();
        if (rL == 0) {
            return 0;
        }
        if (rL < 3) {
            for (size_t i = 0; i < rL; ++i) {
                r3.my_push(r.digitReal(rL - 1 - i));
            }
        } else {
            r3.my_push(r.digitReal(km - 2));
            r3.my_push(r.digitReal(km - 1));
            r3.my_push(r.digitReal(km));
        }
        /*r3.my_push(r.digitReal(km - 3));
        r3.my_push(r.digitReal(km - 2));
        r3.my_push(r.digitReal(km - 1));*/

        //big_integer r3 = (r.digitReal(km - 1) * MAX + r.digitReal(km - 2)) * MAX + r.digitReal(km - 3);

        //big_integer d2 = divider;
        //d2.my_push(d.digitReal(m - 2));
        //d2.my_push(d.digitReal(m - 1));
        //big_integer d2 = d.digitReal(m - 1) * MAX + d.digitReal(m - 2);

        /*big_integer nice;
        nice.my_push(0);
        nice.my_push(0);
        nice.my_push(0);
        nice.my_push(1);

        big_integer _ans = r3 * div_nice(nice, d2);*/
        big_integer _ans = r3 * divider;
        if (_ans.length() >= 4) {
            return _ans.data[3];
        } else {
            return 0;
        }
        /*_ans.my_move();

        unsigned int result = (r3 * div_nice(nice, d2)).my_move().data[2];
        return (r3 * div_nice(nice, d2)).my_move().data[2];*/
    } else {
        return 0;
    }
}

/*
 * function smaller(r, dq: number;
k, m: integer): boolean;
var i, j: integer;
    begin
        {O<=k<=k+m<=w}
        i :== m; j :== 0;
        while i <> j do
            if r[i + k] <> dq[i]
            then j :== i
            else i :== i - 1;
        smaller := r[i + k] < dq[i]
end
 */

bool smaller(big_integer const &r, big_integer const &dq, size_t k) {
    size_t rL = r.length();
    size_t dqL = dq.length();
    if (rL >= dqL) {
        //bool flag = false;
        size_t i = 0;
        for (i = 0; i < dqL - 1; ++i) {
            if (r.data[dqL - 1 - i + k] != dq.data[dqL - 1 - i]) {
                //flag=true;
                break;
            }
        }

        //we wish to check the last symbols

        /*
    while (i != j) {
    if (r.data[i + k] != dq.data[i]) {
        j = i;
    } else {
        i--;
    }
    }*/
        return r.data[dqL - 1 - i + k] < dq.data[dqL - 1 - i];
    } else {
        return true;
    }
}

/*procedure di:fference(var r: number;
dq: number; k, m: integer);
var borrow, di:ff, i: integer;
begin
{0 <= k <= k+m <= w}
borrow:= 0;
for i := 0 to m do
begin
        di:ff := r[i + k] - dq[i]
-borrow+ b;
r[i + k] := di:ff mod b;
borrow := 1 - di:ff div b
        end;
if borrow < > 0 then overflow
end*/

big_integer make_longer(size_t len, big_integer const &a) {
    size_t aL = a.length();
    if (len >= aL) {
        big_integer b;
        b.data.assign(len, 0);
        for (size_t i = 0; i < aL; ++i) {
            b.data[len - 1 - i] = a.data[aL - 1 - i];
        }
        return b;
    } else {
        return a;
    }
}

void difference(big_integer &r, big_integer &dq, size_t k) {
    size_t rL = r.length();
    size_t dqL = dq.length();

    //big_integer r2 = r - make_longer(k + dqL, dq);

    if (rL != 0) {
//        int borrow = 0;
//        size_t rL = r.length();
        //size_t dqL = dq.length();
        long long int sum = castUnsignedLongLong(r.digitReal(k)) + 1ULL + castUnsignedLongLong(~dq.digitReal(0));
        long long int carry = sum >> BASE;
        r.data[k]=sum;
        for (size_t i = 1; i < dqL; ++i) {
            sum = carry + castUnsignedLongLong(r.digitReal(i + k)) + castUnsignedLongLong(~dq.digitReal(i));
            r.data[i + k] = castUnsignedInt(sum);
            carry = sum >> BASE;/*
        diff = r.data[i + k] - dq.data[i] + borrow;
        r.data[i + k] = borrow;
        borrow = diff >> BASE;*/
        }
        r.make_fit();
        /*if (r != r2) {
            r = r2;
        }*/
    }
    /*sum = castUnsignedLongLong(a.digitReal(0)) + 1ULL + castUnsignedLongLong(~b.digitReal(0));
    temp[0] = castUnsignedInt(sum);
    carry = sum >> BASE;
    for (size_t i = 1; i < minLength; ++i) {
        sum = carry + castUnsignedLongLong(a.digitReal(i)) + castUnsignedLongLong(~b.digitReal(i));
        temp[i] = castUnsignedInt(sum);
        carry = sum >> BASE;
    }*/
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
    // нормализация
    const unsigned int f = castUnsignedInt(
            ((unsigned long long) (MAX) + 1) / ((unsigned long long) (abs_b.data.back()) + 1));
    /*const*/ size_t aLength = abs_a.length();
    const size_t bLength = abs_b.length();
    mul_big_small(abs_a.data, abs_a.data, f);
    mul_big_small(abs_b.data, abs_b.data, f);
    abs_a.make_fit();
    abs_b.make_fit();

    const size_t len = aLength - bLength + 1;   // длина результата деления
    //void div_big_small(vector<unsigned int> &res, vector<unsigned int> const &a, const unsigned int b) {
    if (bLength == 1) {
        //const size_t len = aLength - bLength + 1;   // длина результата деления
        //vector<unsigned int> q(len, 0);
        //q = div_big_small(abs_a.data, abs_b.data[0]);
        big_integer res(a.sign ^ b.sign, div_big_small(abs_a.data, abs_b.data[0]));
        res.correct();
        return res;
        //return div_nice(a, b);
    } else {
        //const size_t len = aLength - bLength + 1;   // длина результата деления
        vector<unsigned int> q(len, 0);
        vector<unsigned int> dev(bLength + 1), div(bLength + 1, 0);
        big_integer dq;
        big_integer r = abs_a;
        r.my_push(0);
        //aLength++;
        big_integer d = abs_b;
        big_integer divider = find_divider_pro(d, bLength);
        for (int k = aLength - bLength; k >= 0; k--) {
            if (k + bLength <= aLength) {
                unsigned int qt = trial(r, divider, k, bLength);
                mul_big_small(dq.data, d.data, qt);
                if (!smaller(r, dq + d, k)) {
                    qt = qt + 1;
                    dq = dq + d;
                }
                q[k] = qt;
                difference(r, dq, k);
            }
            //r = r - dq;
            /*product( dq, d, qt );
            if smaller(r, dq, k, m) then
                    begin
            qt := qt- 1;
            product( dq, d, qt)
            end;
            q(k] := qt;
            difference(r, dq, k, m)*/
        }
        //div_big_small(abs_a.data, abs_a.data, f);
        big_integer res(a.sign ^ b.sign, q);
        res.correct();
        return res;
    }
}


/* // КОРРЕКТНЫЙ АЛГОРИТМ ДЕЛЕНИЯ внизу
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
    vector<unsigned int> dev(bLength + 1), div(bLength + 1, 0);
    for (size_t i = 0; i < bLength; i++) {
        dev[i] = abs_a.digitReal(aLength + i - bLength);    // correct because abs_a >= abs_b and aLength >= bLength
    }
    dev[bLength] = abs_a.digit(aLength);    // dev хранит последние bLength+1 ячеек вектора abs_a
    if (bLength == 1) {
        for (size_t i = 0; i < len; i++) {
            dev[0] = abs_a.digitReal(aLength - bLength - i);    // держит i-ую с конца цифру (число 32битное) abs_a
            size_t resPos = len - 1 - i;
            unsigned int smallRes = get_two(dev[bLength], dev[bLength - 1],
                                            divisor); // деление в столбик так работает, две цифры (числа 32битных) - с запасом (иногда как раз)
            mul_big_small(div, abs_b.data,
                          smallRes);   // результат умножения вектора abs_b.data на smallRes в div. В div делимое - остаток от строчки выше
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
            unsigned int smallRes = get_two(dev[bLength], dev[bLength - 1],
                                            divisor); // деление в столбик так работает, два куска по длине делителя - с запасом (иногда как раз)
            mul_big_small(div, abs_b.data,
                          smallRes);   // результат умножения вектора abs_b.data на smallRes в div. В div делимое - остаток от строчки выше
            while ((smallRes >= 0) && compare_equal_vectors(dev, div)) {
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
}*/

big_integer operator%(big_integer const &a, big_integer const &b) {
    return a - (a / b) * b;
}

/*
big_integer c_real("100000000000000000000000000000000000000000000000000000");

int main() {
    big_integer a("948154728221296122");
    big_integer b("250470475485330530");
    //div_nice(a, b);
    //big_integer c  = a*b / b;
    big_integer c1 = div_nice(a * b, a);
    big_integer c2 = (a * b) / a;
    //big_integer c3 = (a * b) / b;
    // std::string str1 = to_string(c);
    std::string str2 = to_string(c_real);
    return 0;
}*/

//в статье из 3 в 2