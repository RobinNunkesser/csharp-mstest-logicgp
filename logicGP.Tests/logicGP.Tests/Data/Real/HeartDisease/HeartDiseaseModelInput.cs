using Microsoft.ML.Data;

namespace logicGP.Tests.Data.Real;

public class HeartDiseaseModelInput
{
    [LoadColumn(0)] [ColumnName(@"age")] public float Age { get; set; }

    [LoadColumn(1)] [ColumnName(@"sex")] public float Sex { get; set; }

    [LoadColumn(2)] [ColumnName(@"cp")] public float Cp { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"trestbps")]
    public float Trestbps { get; set; }

    [LoadColumn(4)] [ColumnName(@"chol")] public float Chol { get; set; }

    [LoadColumn(5)] [ColumnName(@"fbs")] public float Fbs { get; set; }

    [LoadColumn(6)]
    [ColumnName(@"restecg")]
    public float Restecg { get; set; }

    [LoadColumn(7)]
    [ColumnName(@"thalach")]
    public float Thalach { get; set; }

    [LoadColumn(8)] [ColumnName(@"exang")] public float Exang { get; set; }

    [LoadColumn(9)]
    [ColumnName(@"oldpeak")]
    public float Oldpeak { get; set; }

    [LoadColumn(10)]
    [ColumnName(@"slope")]
    public float Slope { get; set; }

    [LoadColumn(11)] [ColumnName(@"ca")] public float Ca { get; set; }

    [LoadColumn(12)] [ColumnName(@"thal")] public float Thal { get; set; }

    [LoadColumn(13)] [ColumnName(@"num")] public uint Num { get; set; }
}

public class HeartDiseaseModelInputOriginal
{
    [LoadColumn(0)] [ColumnName(@"age")] public float Age { get; set; }

    [LoadColumn(1)] [ColumnName(@"sex")] public float Sex { get; set; }

    [LoadColumn(2)] [ColumnName(@"cp")] public float Cp { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"trestbps")]
    public float Trestbps { get; set; }

    [LoadColumn(4)] [ColumnName(@"chol")] public float Chol { get; set; }

    [LoadColumn(5)] [ColumnName(@"fbs")] public float Fbs { get; set; }

    [LoadColumn(6)]
    [ColumnName(@"restecg")]
    public float Restecg { get; set; }

    [LoadColumn(7)]
    [ColumnName(@"thalach")]
    public float Thalach { get; set; }

    [LoadColumn(8)] [ColumnName(@"exang")] public float Exang { get; set; }

    [LoadColumn(9)]
    [ColumnName(@"oldpeak")]
    public float Oldpeak { get; set; }

    [LoadColumn(10)]
    [ColumnName(@"slope")]
    public float Slope { get; set; }

    [LoadColumn(11)] [ColumnName(@"ca")] public float Ca { get; set; }

    [LoadColumn(12)] [ColumnName(@"thal")] public float Thal { get; set; }

    [LoadColumn(13)] [ColumnName(@"num")] public float Num { get; set; }
}