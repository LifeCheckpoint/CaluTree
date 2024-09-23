# wolfram 代码中转
from opt import *

if opt.wolfram_using:
    from wolframclient.evaluation import WolframLanguageSession
    from wolframclient.language import wl, wlexpr

class wolfram:
    # 执行wolfram语句
    def wolfram_evaluate(self, expr):
        return self.session.evaluate(wlexpr(expr))

    def stop(self):
        self.session.stop()

    # 初始化wolfram语句，支持算法运行
    def __init__(self):
        # 创建wolfram会话
        self.session = WolframLanguageSession(kernel=opt.wolfram_path)

        # 符号替换初始化
        symbol_init = """
        symbolList=Hold@{Pi,E,Pi/E,E/Pi,Pi+E,Pi-E,Pi E,E^Pi,Pi^E,Pi Pi,E E,Pi+Pi,E+E,Pi^Pi,E^E,(Pi Pi Pi)^Pi/(Pi-E)};
        symbolListHold=symbolList/.{s_Symbol[a_,b_]->Hold[s[a,b]]}//ReleaseHold;
        replaceSymbol=AssociationThread[ToExpression["#"<>ToString[#]&/@Range[Length[symbolListHold]]]->symbolListHold];
        """

        # 树计算处理初始化
        tree_init = """
        SetAttributes[ReplaceExpr,HoldAll];
        ReplaceExpr[expr_]:=Hold[expr]/.replaceSymbol;
        ExprRelease[holdExpr_]:=Nest[ReleaseHold,holdExpr,Max[Length/@Position[holdExpr,Hold]]];
        ExprN[holdExpr_,precision_]:=N[ExprRelease[holdExpr],precision];
        Hold2HoldForm[holdExpr_]:=holdExpr/.{Hold->HoldForm};
        """

        # 微分引擎初始化
        diff_init = """
        DeltaTest[expr_]:=Block[{posPi,posE,repsPi,repsE,deltaPi,deltaE},
        posPi=Position[expr,Pi];
        posE=Position[expr,E];
        repsPi=ReplaceAt[expr,Pi->x,#]&/@posPi;repsE=ReplaceAt[expr,E->x,#]&/@posE;
        deltaPi={#,D[ExprRelease[#],x]/.{x->Pi}//N}&/@repsPi//DeleteCases[{_,x_/;x==0}];
        deltaE={#,D[ExprRelease[#],x]/.{x->E}//N}&/@repsE//DeleteCases[{_,x_/;x==0}];
        Return[ReverseSortBy[deltaPi~Join~deltaE,Last]];
        ];
        """
        
        if self.wolfram_evaluate(symbol_init) is not None:
            print("A error occured when Wolfram code 'symbol' loading")
        if self.wolfram_evaluate(tree_init) is not None:
            print("A error occured when Wolfram code 'symbol' loading")
        if self.wolfram_evaluate(diff_init) is not None:
            print("A error occured when Wolfram code 'symbol' loading")

# 测试用例
if __name__ == "__main__":
    wolf = wolfram()
    eva = wolf.wolfram_evaluate("N[{Pi, E, Pi/E, E/Pi, Pi+E, Pi-E, Pi E, E^Pi, Pi^E, Pi Pi, E E, Pi+Pi, E+E, Pi^Pi, E^E},17]")
    print([float(i) for i in list(eva)])
    wolf.stop()