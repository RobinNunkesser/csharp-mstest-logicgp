using Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;

namespace logicGP.Tests.Unit.Search.GP.SearchSpace;

[TestClass]
public class LogicGpLiteralTests
{
    private readonly List<LogicGpLiteral<string>> _literals = [];

    public required List<string> TrainingData =
    [
        "1",
        "2",
        "3"
    ];

    public required HashSet<string> UniqueValues =
    [
        "1",
        "2",
        "3"
    ];

    [TestMethod]
    public void TestLogicGpAllSuLiterals()
    {
        var set = 0;
        var negativeset = 0;
        for (var i = 1; i <= UniqueValues.Count; i++)
        {
            for (var j = i; j <= UniqueValues.Count; j++)
            {
                if (i == 1 && j == UniqueValues.Count)
                    continue;
                set = set + (1 << (j - 1));
                negativeset = ~set & ((1 << UniqueValues.Count) - 1);
                var literal = new LogicGpLiteral<string>(0, "SNP",
                    UniqueValues, set,
                    TrainingData,
                    LogicGpLiteralType.Su);
                if (!_literals.Contains(literal))
                    _literals.Add(literal);
                var negativeliteral = new LogicGpLiteral<string>(0, "SNP",
                    UniqueValues, negativeset,
                    TrainingData,
                    LogicGpLiteralType.Su);
                if (!_literals.Contains(negativeliteral))
                    _literals.Add(negativeliteral);
            }

            set = 0;
            negativeset = 0;
        }

        var expectedLiterals = new List<string>
        {
            "(SNP ∈ [1,1])",
            "(SNP ∈ [2,3])",
            "(SNP ∈ [1,2])",
            "(SNP ∈ [3,3])",
            "(SNP ∈ [2,2])",
            "(SNP ∉ [2,2])"
        };
        var actualLiterals = _literals.Select(l => l.ToString()).ToList();
        Assert.AreEqual(expectedLiterals.Count, actualLiterals.Count);
        for (var i = 0; i < expectedLiterals.Count; i++)
            Assert.AreEqual(expectedLiterals[i], actualLiterals[i]);
    }

    [TestMethod]
    public void TestLogicGpAllLessGreaterLiterals()
    {
        var set = 0;
        var negativeset = 0;
        for (var i = 1; i < UniqueValues.Count; i++)
        {
            set = set + (1 << (i - 1));
            negativeset = negativeset + (1 << (UniqueValues.Count - i));

            var literal = new LogicGpLiteral<string>(0, "SNP",
                UniqueValues, set,
                TrainingData,
                LogicGpLiteralType.LessGreater);
            _literals.Add(literal);
            var negativeliteral = new LogicGpLiteral<string>(0, "SNP",
                UniqueValues, negativeset,
                TrainingData,
                LogicGpLiteralType.LessGreater);
            _literals.Add(negativeliteral);
        }

        var expectedLiterals = new List<string>
        {
            "(SNP < 2)",
            "(SNP > 2)",
            "(SNP < 3)",
            "(SNP > 1)"
        };
        var actualLiterals = _literals.Select(l => l.ToString()).ToList();
        Assert.AreEqual(expectedLiterals.Count, actualLiterals.Count);
        for (var i = 0; i < expectedLiterals.Count; i++)
            Assert.AreEqual(expectedLiterals[i], actualLiterals[i]);
    }

    [TestMethod]
    public void TestLogicGpAllDussaultLiterals()
    {
        for (var i = 1; i <= UniqueValues.Count; i++)
        {
            var set = 1 << (i - 1);
            var negativeset = ~set & ((1 << UniqueValues.Count) - 1);
            var literal = new LogicGpLiteral<string>(0, "SNP",
                UniqueValues, set,
                TrainingData,
                LogicGpLiteralType.Dussault);
            _literals.Add(literal);
            var negativeliteral = new LogicGpLiteral<string>(0, "SNP",
                UniqueValues, negativeset,
                TrainingData,
                LogicGpLiteralType.Dussault);
            _literals.Add(negativeliteral);
        }

        var expectedLiterals = new List<string>
        {
            "(SNP = 1)",
            "(SNP ≠ 1)",
            "(SNP = 2)",
            "(SNP ≠ 2)",
            "(SNP = 3)",
            "(SNP ≠ 3)"
        };
        var actualLiterals = _literals.Select(l => l.ToString()).ToList();
        Assert.AreEqual(expectedLiterals.Count, actualLiterals.Count);
        for (var i = 0; i < expectedLiterals.Count; i++)
            Assert.AreEqual(expectedLiterals[i], actualLiterals[i]);
    }

    [TestMethod]
    public void TestLogicGpAllRudellLiterals()
    {
        var powerSetCount = 1 << UniqueValues.Count;
        for (var i = 1; i < powerSetCount - 1; i++)
        {
            var literal = new LogicGpLiteral<string>(0, "SNP",
                UniqueValues, i, TrainingData);
            _literals.Add(literal);
        }

        var expectedLiterals = new List<string>
        {
            "(SNP ∈ {1})",
            "(SNP ∈ {2})",
            "(SNP ∈ {1,2})",
            "(SNP ∈ {3})",
            "(SNP ∈ {1,3})",
            "(SNP ∈ {2,3})"
        };
        var actualLiterals = _literals.Select(l => l.ToString()).ToList();
        Assert.AreEqual(expectedLiterals.Count, actualLiterals.Count);
        for (var i = 0; i < expectedLiterals.Count; i++)
            Assert.AreEqual(expectedLiterals[i], actualLiterals[i]);
    }
}