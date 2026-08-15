// ==============================================================================
// Copyright (c) 2026 Muhamad Syahrul Hidayat and mshqc contributors
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
// ==============================================================================
func.func @test_mp2_contraction(%lhs: tensor<10x20x30xf64>, %rhs: tensor<10x20x30xf64>) -> tensor<10x20x10x20xf64> {
  %res = mshqc.contract %lhs, %rhs {einsum_eq = "iaP,jbP->iajb"} : (tensor<10x20x30xf64>, tensor<10x20x30xf64>) -> tensor<10x20x10x20xf64>
  return %res : tensor<10x20x10x20xf64>
}