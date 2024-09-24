# wolfram 代码中转
import os
import traceback
import multiprocessing as mp
from colorama import Fore
from decimal import Decimal
from opt import *

if opt.enable_wolfram and mp.current_process().name == "MainProcess":
    from wolframclient.evaluation import WolframLanguageSession
    from wolframclient.language import wlexpr

# 定义的常数池，参与运算搜索
consts_pool = "Pi, E, Pi/E, E/Pi, Pi+E, Pi-E, Pi E, E^Pi, Pi^E, Pi Pi, E E, Pi+Pi, E+E, Pi^Pi, E^E, (Pi Pi Pi)^Pi/(Pi-E)"

class wolfram:
    # 执行wolfram语句
    def wolfram_evaluate(self, expr):
        try:
            return self.session.evaluate(wlexpr(expr))
        except Exception as e:
            print(e)
            traceback.print_exc()
            return ""

    def stop(self):
        self.session.stop()

    # 初始化wolfram语句，支持算法运行
    def __init__(self):
        # 创建wolfram会话
        self.session = WolframLanguageSession(kernel=opt.wolfram_path)

        # 符号替换初始化
        symbol_init = """
        symbolList=Hold@{CONSTS};
        symbolListHold=symbolList/.{s_Symbol[a_,b_]->Hold[s[a,b]]}//ReleaseHold;
        replaceSymbol=AssociationThread[ToExpression["#"<>ToString[#]&/@Range[Length[symbolListHold]]]->symbolListHold];
        """.replace("CONSTS", consts_pool)

        # 树计算处理初始化
        tree_init = """
        SetAttributes[ReplaceExpr,HoldAll];
        ReplaceExpr[expr_]:=Hold[expr]/.replaceSymbol;
        ExprRelease[holdExpr_]:=Block[{layer},layer=Max[Length/@Position[holdExpr,Hold]];
        Return[If[layer!=-Infinity,Nest[ReleaseHold,holdExpr,layer],holdExpr]];
        ];
        ExprN[holdExpr_,precision_]:=N[ExprRelease[holdExpr],precision]//Re;
        Hold2HoldForm[holdExpr_]:=holdExpr/.{Hold->HoldForm};
        InverseSolve[expr_,target_]:=Quiet[Last@First@First@N[FindInstance[ExprRelease[expr]==target,x,Reals,WorkingPrecision->30],30]];
        """

        # 微分引擎初始化
        diff_init = """
        DeltaTest[expr_]:=Block[{posPi,posE,repsPi,repsE,deltaPi,deltaE},
        posPi=Position[expr,Pi];
        posE=Position[expr,E];
        repsPi=ReplaceAt[expr,Pi->x,#]&/@posPi;repsE=ReplaceAt[expr,E->x,#]&/@posE;
        deltaPi={#,D[ExprRelease[#],x]/.{x->Pi}//N}&/@repsPi//DeleteCases[{_,x_/;x==0}];
        deltaE={#,D[ExprRelease[#],x]/.{x->E}//N}&/@repsE//DeleteCases[{_,x_/;x==0}];
        Return[SortBy[deltaPi~Join~deltaE,Abs@Last@#&]];
        ];
        """
       
        if self.wolfram_evaluate(symbol_init) is not None:
            self.err("symbol")
        if self.wolfram_evaluate(tree_init) is not None:
            self.err("tree")
        if self.wolfram_evaluate(diff_init) is not None:
            self.err("diff")

    # 精确数值
    def N(self, expr: str, precision=20) -> Decimal:
        return self.wolfram_evaluate(f"N[{expr},{precision}]")
    
    def err(self, err_part):
        print(f"A error occured when Wolfram code '{err_part}' loading")
        print(f"{Fore.RED}Wolfram 唤起失败，请重新执行程序.{Fore.WHITE}")
        os._exit(-1)

# 测试用例
if __name__ == "__main__":
    wolf = wolfram()
    eva = wolf.wolfram_evaluate("N[{Pi, E, Pi/E, E/Pi, Pi+E, Pi-E, Pi E, E^Pi, Pi^E, Pi Pi, E E, Pi+Pi, E+E, Pi^Pi, E^E},17]")
    # print([float(i) for i in list(eva)])
    print(wolf.N("Pi/E"))
    wolf.stop()