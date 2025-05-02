using Microsoft.ML.Data;

namespace logicGP.Tests.Data.Real;

public class BalanceScaleModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"right-distance")]
    public float Right_distance { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"right-weight")]
    public float Right_weight { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"left-distance")]
    public float Left_distance { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"left-weight")]
    public float Left_weight { get; set; }

    [LoadColumn(4)] [ColumnName(@"class")] public string Class { get; set; }
}