// Copyright 2026 Muhamad Syahrul Hidayat
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#ifdef _MSC_VER
#include <bit>
#include <cstdint>

#ifndef __builtin_clz
inline int __builtin_clz(unsigned int x) noexcept {
    return static_cast<int>(std::countl_zero(x));
}
#endif

#ifndef __builtin_clzll
inline int __builtin_clzll(unsigned long long x) noexcept {
    return static_cast<int>(std::countl_zero(x));
}
#endif

#ifndef __builtin_ctz
inline int __builtin_ctz(unsigned int x) noexcept {
    return static_cast<int>(std::countr_zero(x));
}
#endif

#ifndef __builtin_popcount
inline int __builtin_popcount(unsigned int x) noexcept {
    return static_cast<int>(std::popcount(x));
}
#endif

#ifndef __builtin_expect
#define __builtin_expect(expr, val) (expr)
#endif
#endif