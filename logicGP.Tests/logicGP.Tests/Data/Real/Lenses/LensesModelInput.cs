using Microsoft.ML.Data;

namespace logicGP.Tests.Data.Real;

public class LensesModelInput
{
    [LoadColumn(0)] [ColumnName(@"age")] public float Age { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"spectacle_prescription")]
    public float Spectacle_prescription { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"astigmatic")]
    public float Astigmatic { get; set; }

    [LoadColumn(3)] [ColumnName(@"class")] public uint Class { get; set; }
}