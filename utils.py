def hasTradeStat(test):
    testA = test.loc[test["has_trade"]==1]
    ratio = len(testA.loc[testA["is_trade"] == 1].values) * 1.0 / len(testA.values)
    testB = test.loc[test["has_trade"]==0]
    ratio2 = len(testB.loc[testB["is_trade"] == 1].values) * 1.0 / len(testB.values)
    print("has {0} hasnot {1}".format(ratio,ratio2))
