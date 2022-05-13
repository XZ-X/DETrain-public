//
// Created by xuxiangzhe on 10/18/2021.
//
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
//#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/InstVisitor.h>
//#include <llvm/IR/Metadata.h>
//#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
//#include <llvm/IRReader/IRReader.h>
//#include <llvm/CodeGen/IntrinsicLowering.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <unordered_map>
#include <unordered_set>

//#define DBG

using namespace llvm;
using namespace std;

namespace {
using Definition = unordered_map<Value *, unordered_set<Value *>>;

struct CallAnalysis : public InstVisitor<CallAnalysis, void> {
  Definition &defines, &alias;
  static unordered_set<Value *> fakeValue;
  unordered_set<string> ret;

  CallAnalysis(Definition &defines, Definition &alias)
      : defines(defines), alias(alias) {}

  bool update = false;
  bool collectMode;

  uint64_t getOfsFromGEP(const DataLayout &dl, gep_type_iterator begin,
                         gep_type_iterator end) {
    uint64_t ofs = 0;
    for (auto t = begin; t != end; t++) {
      if (auto sType = t.getStructTypeOrNull()) {
        auto sLayout = dl.getStructLayout(sType);
        auto idx = cast<ConstantInt>(t.getOperand())->getZExtValue();
        ofs += sLayout->getElementOffset(idx);
      } else {
        // Not constant! Give up!
        return 0;
      }
    }
    return ofs;
  }

  // Value *getValueFromConstantExpr(ConstantExpr &expr, Module &m) {
  //     switch (expr.getOpcode()) {
  //         default:
  //             return nullptr;
  //         case Instruction::GetElementPtr: {
  //             auto ofs = getOfsFromGEP(m.getDataLayout(),
  //             gep_type_begin(expr), gep_type_end(expr)); uint64_t base =
  //             (uint64_t) expr.getOperand(0);
  //             // hacking
  //             auto ret = (Value *) ofs + base;
  //             if (ofs != 0) {
  //                 fakeValue.insert(ret);
  //             }
  //             return ret;
  //         }
  //     }
  // }
  void visitCallBase(CallBase& I){

    auto callee = I.getCalledOperand();
    if (auto func = dyn_cast<Function>(callee)) {
      //                outs() << "CALL FUNC " << func->getName() << "\n";
      ret.insert(func->getName());
    } else {
      //                if (defines.count(callee)) {
      //                    outs() << "CALL FP: ";
      for (auto v : defines[callee]) {
        if (fakeValue.count(v)) {
          continue;
        }
        if (v->getType()->isPointerTy() &&
            v->getType()->getContainedType(0)->isFunctionTy()) {
          auto f = v->stripPointerCasts();
          //                            outs() << f->getName() << ",";
          ret.insert(f->getName());
        }
      }
      for (auto a : alias[callee]) {
        if (fakeValue.count(a)) {
          continue;
        }
        if (a->getType()->isPointerTy() &&
            a->getType()->getContainedType(0)->isFunctionTy()) {
          auto f = a->stripPointerCasts();
          //                            outs() << f->getName() << ",";
          ret.insert(f->getName());
        }
      }
      //                    outs() << "\n";
      //                }
      //                outs() << "\n";
    }
  }

  void visitInvoke(InvokeInst &I){
    if (!collectMode) {
      return;
    }
    visitCallBase(I);
  }

  void visitCall(CallInst &I) {
    if (!collectMode) {
      return;
    }
    visitCallBase(I);
  }

  void visitLoad(LoadInst &I) {
    auto from = I.getPointerOperand();
    if (auto constExpr = dyn_cast<ConstantExpr>(from)) {
      const auto &m = *I.getParent()->getParent()->getParent();
      // if (auto gep = getValueFromConstantExpr(*constExpr,
      // *I.getParent()->getParent()->getParent())) {
      //     from = gep;
      // }
      if (Instruction::GetElementPtr == constExpr->getOpcode()) {
        auto ofs = getOfsFromGEP(m.getDataLayout(), gep_type_begin(constExpr),
                                 gep_type_end(constExpr));
        uint64_t base = (uint64_t)constExpr->getOperand(0);
        auto adjustedValue = (Value *)(base + ofs);

        if (ofs != 0) {
          fakeValue.insert(adjustedValue);
        }
        alias[&I].insert(defines[adjustedValue].begin(),
                         defines[adjustedValue].end());

        for (auto a : alias[constExpr->getOperand(0)]) {
          auto base = (uint64_t)a;
          auto adjustedValue = (Value *)(a + ofs);
          if (ofs != 0) {
            fakeValue.insert(adjustedValue);
          }
          alias[&I].insert(defines[adjustedValue].begin(),
                           defines[adjustedValue].end());
        }

        return;
      }
    }

    if (defines.count(from)) {
      alias[&I].insert(defines[from].begin(), defines[from].end());
    }
    for (auto a : alias[from]) {
      if (defines.count(a)) {
        alias[&I].insert(defines[a].begin(), defines[a].end());
      }
    }
  }

  void visitStore(StoreInst &I) {
    auto value = I.getValueOperand();
    auto to = I.getPointerOperand();
    unordered_set<Value *> defs;
    if (alias.count(value)) {
      defs = alias[value];
    }
    defs.insert(value);

    bool isGEP = false;
    if (auto constExpr = dyn_cast<ConstantExpr>(to)) {
      const auto &m = *I.getParent()->getParent()->getParent();
      if (Instruction::GetElementPtr == constExpr->getOpcode()) {
        auto ofs = getOfsFromGEP(m.getDataLayout(), gep_type_begin(constExpr),
                                 gep_type_end(constExpr));
        uint64_t base = (uint64_t)constExpr->getOperand(0);
        auto adjustedValue = (Value *)(base + ofs);

        if (ofs != 0) {
          fakeValue.insert(adjustedValue);
        }
        defines[adjustedValue].clear();
        defines[adjustedValue].insert(defs.begin(), defs.end());
        // alias[&I].insert(defines[adjustedValue].begin(),defines[adjustedValue].end());

        for (auto a : alias[constExpr->getOperand(0)]) {
          auto base = (uint64_t)a;
          auto adjustedValue = (Value *)(a + ofs);
          if (ofs != 0) {
            fakeValue.insert(adjustedValue);
          }
          defines[adjustedValue].clear();
          defines[adjustedValue].insert(defs.begin(), defs.end());
          // alias[&I].insert(defines[adjustedValue].begin(),defines[adjustedValue].end());
        }

        return;
      }
    }

    //            if(!to->getType()->getContainedType(0)->isAggregateType()){
    defines[to].clear();
    //            }

    defines[to].insert(defs.begin(), defs.end());
    for (auto a : alias[to]) {
      defines[a].insert(defs.begin(), defs.end());
    }
  }

  void visitGetElementPtrInst(GetElementPtrInst &I) {
    auto from = I.getPointerOperand();
    const auto module = I.getParent()->getParent()->getParent();
    auto ofs = getOfsFromGEP(module->getDataLayout(), gep_type_begin(I),
                             gep_type_end(I));
    auto adjustedValue = (Value *)((uint64_t)from + ofs);
    if (ofs != 0) {
      fakeValue.insert(adjustedValue);
    }

    alias[&I] = alias[adjustedValue];
    defines[&I] = defines[adjustedValue];
    for (auto a : alias[from]) {
      auto adjustedValue = (Value *)((uint64_t)a + ofs);
      if (ofs != 0) {
        fakeValue.insert(adjustedValue);
      }
      alias[&I].insert(alias[adjustedValue].begin(),
                       alias[adjustedValue].end());
      defines[&I].insert(defines[adjustedValue].begin(),
                         defines[adjustedValue].end());
    }
  }
};
unordered_set<Value *> CallAnalysis::fakeValue = unordered_set<Value *>();

struct CGraph : public ModulePass {
  static char ID;
  CGraph() : ModulePass(ID) {}

  //    struct CGraph {
  // o1 += o2
  void merge(Definition &o1, Definition &o2) {
    for (auto valueDef : o2) {
      if (o1.count(valueDef.first)) {
        o1[valueDef.first].insert(valueDef.second.begin(),
                                  valueDef.second.end());
      } else {
        o1[valueDef.first] = valueDef.second;
      }
    }
    o1.insert(o2.begin(), o2.end());
  }

  bool runOnModule(Module &M) {
    unordered_map<BasicBlock *, Definition> beforeDef, afterDef;
    unordered_map<BasicBlock *, Definition> beforeAlias, afterAlias;
    for (auto &F : M) {
      auto change = true;
      while (change) {
        change = false;
        for (auto &B : F) {
          Definition def, alias;
          for (auto pred : predecessors(&B)) {
            merge(def, afterDef[pred]);
            merge(alias, afterAlias[pred]);
          }
          {
            auto &prevD = beforeDef[&B];
            if ((!change) && prevD != def) {
              change = true;
            }
            beforeDef[&B] = def;

            auto &prevA = beforeAlias[&B];
            if ((!change) && prevA != alias) {
              change = true;
            }
            beforeAlias[&B] = alias;
          }
          CallAnalysis visitor(def, alias);
          for (auto &I : B) {
#ifdef DBG
            outs() << "VISIT:::::" << I << "\n";
#endif
            visitor.visit(I);
#ifdef DBG
            outs() << "ALIAS: \n";
            for (auto &aa : alias) {
              aa.first->printAsOperand(outs());
              outs() << ": ";
              for (auto v : aa.second) {
                v->printAsOperand(outs());
                outs() << ",";
              }
              outs() << "\n";
            }
            outs() << "VDEF: \n";
            for (auto &vd : def) {
              vd.first->printAsOperand(outs());
              outs() << ": ";
              for (auto d : vd.second) {
                d->printAsOperand(outs());
                outs() << ",";
              }
              outs() << "\n";
            }
#endif
          }
          {
            auto &prevD = afterDef[&B];
            if ((!change) && prevD != def) {
              change = true;
            }
            afterDef[&B] = def;

            auto &prevA = afterAlias[&B];
            if ((!change) && prevA != alias) {
              change = true;
            }
            afterAlias[&B] = alias;
          }
        }
      }
#ifdef DBG
      outs() << "BEGIN " << F.getName() << ":\n";
#endif
      unordered_set<string> ret;
      for (auto &B : F) {
        auto def = beforeDef[&B];
        auto alias = beforeAlias[&B];
        CallAnalysis visitor(def, alias);
        visitor.collectMode = true;
        for (auto &I : B) {
          visitor.visit(I);
        }
        ret.insert(visitor.ret.begin(), visitor.ret.end());
      }
      for (auto &s : ret) {
        if (s.empty()) {
          continue;
        }
        if (boost::algorithm::contains(s, "ReserveRandomOutputs")) {
          outs() << F.getName() << ":"
                 << " " << s << endl;
        }
      }
      // outs() << "\n";
#ifdef DBG
      outs() << "END " << F.getName() << ".\n";
#endif
    }
    outs().flush();
    errs().flush();
    return false;
  }
};
} // namespace

// void analyze(Module &m) {
//     ::CGraph cgPass;
//     cgPass.runOnModule(m);
// }

char CGraph::ID = '2';

static RegisterPass<CGraph> X("cgraph", "Call Graph Pass", false, false);

static void registerCGraphPass(const PassManagerBuilder &,
                               legacy::PassManagerBase &PM) {
  PM.add(new CGraph());
}

static RegisterStandardPasses
    RegisterCGraphPass(PassManagerBuilder::EP_ModuleOptimizerEarly,
                       registerCGraphPass);

static RegisterStandardPasses
    RegisterCGraphPass0(PassManagerBuilder::EP_EnabledOnOptLevel0,
                        registerCGraphPass);