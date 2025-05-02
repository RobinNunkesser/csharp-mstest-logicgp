using System.Text.Json.Serialization;

namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

public class TabularFileDataSourceV3 : MBConfig, ITabularFileDataSource
{
    public override int Version => 3;
    public override string Type => "TabularFile";

    [JsonIgnore] public DataSourceType DataSourceType { get; set; }

    public IEnumerable<IColumnProperties> ColumnProperties { get; set; }
    public string? FilePath { get; set; }
    public string? Delimiter { get; set; }

    public char DecimalMarker { get; set; }

    public bool HasHeader { get; set; }
    public bool AllowQuoting { get; set; }
    public char EscapeCharacter { get; set; }
    public bool ReadMultiLines { get; set; }
    [JsonIgnore] public bool KeepDiacritics { get; set; }
    [JsonIgnore] public bool KeepPunctuations { get; set; }
}