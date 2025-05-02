namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

public class ColumnPropertiesV5 : MBConfig, IColumnProperties
{
    public override int Version => 5;
    public override string Type => "Column";
    public string? ColumnName { get; set; }
    public ColumnPurposeType ColumnPurpose { get; set; }
    public ColumnDataKind ColumnDataFormat { get; set; }
    public bool IsCategorical { get; set; }
}