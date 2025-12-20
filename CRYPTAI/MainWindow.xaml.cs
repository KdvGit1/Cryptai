using CRYPTAI.Helpers;
using CRYPTAI.Models;
using CRYPTAI.Services;
using ScottPlot;
using System.Net.Http;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using Application = System.Windows.Application;
using Binding = System.Windows.Data.Binding;
using Brushes = System.Windows.Media.Brushes;
using Color = System.Windows.Media.Color;
using ColorConverter = System.Windows.Media.ColorConverter;
using Image = System.Windows.Controls.Image;
using MouseEventArgs = System.Windows.Input.MouseEventArgs;

namespace CRYPTAI
{
    

    public partial class MainWindow : Window
    {
        // --- SERVİSLER ---
        private readonly PythonBackendService _backendService;
        private readonly MarketDataService _marketService;

        // --- UI DEĞİŞKENLERİ ---
        private List<CoinItem> _allCoinsBackup = new();
        private readonly HttpClient _chartHttpClient = new();

        public MainWindow()
        {
            InitializeComponent();

            _marketService = new MarketDataService();
            _backendService = new PythonBackendService();

            _backendService.OnLogReceived += AppendToTerminal;
            _backendService.OnRecoveryNeeded += ShowRepairDialog;
            SetupDataGrids();

            if (HomePAGEText != null) HomePAGEText.Opacity = 0;
            Task.Run(() => _backendService.StartBackend());
            this.Closing += MainWindow_Closing;
            Application.Current.Exit += (s, e) => _backendService.KillProcess();
            ShowStartupBanner();
        }

        // --- YARDIMCI METOT: ALERT GÖSTER ---
        private void ShowAlertDialog(string title, string message, bool isError = false)
        {
            Dispatcher.Invoke(() =>
            {
                var alert = new AlertDialog(title, message, isError);
                alert.Owner = this;
                alert.ShowDialog();
            });
        }

        // ---  ONARIM PENCERESİ  ---
        private void ShowRepairDialog()
        {
            Dispatcher.Invoke(() =>
            {
                _backendService.KillProcess();
                RepairDialog dialog = new RepairDialog();
                dialog.Owner = this;
                dialog.ShowDialog();

                if (dialog.IsRepairRequested)
                {
                    AppendToTerminal("KULLANICI: Onarım talep edildi. Sistem onarılıyor...");
                    Properties.Settings.Default.IsDependenciesInstalled = false;
                    Properties.Settings.Default.Save();
                    Task.Run(() => _backendService.StartBackend(forceCheck: true));
                }
                else
                {
                    AppendToTerminal("KULLANICI: Onarım reddedildi. Sistem kararsız.");
                }
            });
        }

        private async void MainWin_Loaded(object sender, RoutedEventArgs e)
        {
            Sagpanel.Width = new GridLength(9.2, GridUnitType.Star);
            Solpanel.Width = new GridLength(0.8, GridUnitType.Star);
            UpdateSentiment(50);
            if (LoadingGrid is not null) LoadingGrid.Visibility = Visibility.Visible;
            await RefreshMarketData(retryUntilSuccess: true);
            if (LoadingGrid is not null) LoadingGrid.Visibility = Visibility.Collapsed;
        }

        // --- VERİ GÜNCELLEME ---
        private async Task RefreshMarketData(bool retryUntilSuccess)
        {
            bool success = false;
            int retryCount = 0;

            do
            {
                try
                {
                    var data = await _marketService.GetMarketDataAsync();
                    _allCoinsBackup = data;
                    ApplyFilters();
                    success = true;
                }
                catch (Exception ex)
                {
                    if (retryUntilSuccess)
                    {
                        await Task.Delay(2000);
                        retryCount++;
                        if (retryCount > 30) break;
                    }
                    else
                    {
                        AppendToTerminal($"VERİ HATASI: {ex.Message}");
                        break;
                    }
                }
            } while (retryUntilSuccess && !success);
        }

        private void ApplyFilters()
        {
            if (_allCoinsBackup is null || _allCoinsBackup.Count == 0) return;

            string searchText = SearchText.Text.Trim();
            var query = _allCoinsBackup.AsEnumerable();

            if (!string.IsNullOrEmpty(searchText))
            {
                query = query.Where(x => x.Symbol.StartsWith(searchText, StringComparison.OrdinalIgnoreCase));
            }

            UpdateGrids(query.ToList());
        }

        private void UpdateGrids(List<CoinItem> sourceList)
        {
            MyDataGrid.ItemsSource = sourceList.OrderByDescending(x => Math.Abs(x.AiPercentage)).ToList();
        }

        public void UpdateSentiment(double longPercentage)
        {
            if (longPercentage > 100) longPercentage = 100;
            if (longPercentage < 0) longPercentage = 0;

            double shortPercentage = 100 - longPercentage;

            Dispatcher.Invoke(() =>
            {
                if (ColBull != null) ColBull.Width = new GridLength(longPercentage, GridUnitType.Star);
                if (ColBear != null) ColBear.Width = new GridLength(shortPercentage, GridUnitType.Star);

                if (TxtLongRate != null) TxtLongRate.Text = $"%{longPercentage:0.0}";
                if (TxtShortRate != null) TxtShortRate.Text = $"%{shortPercentage:0.0}";
                if (SentimentBorder != null) SentimentBorder.UpdateLayout();
            });
        }

        private void MyDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (MyDataGrid.SelectedItem is CoinItem selectedCoin)
            {
                double direction = selectedCoin.AiPercentage >= 0 ? 1.0 : -1.0;
                double risk = 50.0; 
                string riskString = selectedCoin.RiskRatio?.ToString().Replace("%", "").Trim();
                if (double.TryParse(riskString, out double parsedRisk)) {risk = parsedRisk; }
                if (risk < 0) risk = 0;
                if (risk > 100) risk = 100;
                double movementPower = 100.0 - risk;
                double deviation = movementPower / 2.0;
                double calculatedSentiment = 50 + (deviation * direction);
                UpdateSentiment(calculatedSentiment);
            }
        }
        private void ShowStartupBanner()
        {
            string banner = @"
_________________________________________________

      .---.                  CRYPTAI  
     /   _ \        INTELLIGENT TRADING SYSTEM
 o--|  |_) |      v1.0.4 - Stable Build
    |   _ <       Powered by: Python AI & .NET 8
 o--|  |_) |      
     \_____/         SYSTEM STATUS: [ONLINE]
                     MODE: AUTOMATIC
_________________________________________________
";

            Dispatcher.Invoke(() =>
            {
                TerminalOutput.AppendText(banner + "\n");
                TerminalOutput.ScrollToEnd();
            });
        }
        private void SetupDataGrids()
        {
            var grids = new[] { MyDataGrid };

            for (int i = 0; i < grids.Length; i++)
            {
                var grid = grids[i];
                grid.IsReadOnly = true;
                grid.AutoGenerateColumns = false;
                grid.Columns.Clear();
                grid.GridLinesVisibility = DataGridGridLinesVisibility.None;
                grid.RowHeight = 35;

                var leftStyle = new Style(typeof(TextBlock));
                leftStyle.Setters.Add(new Setter(TextBlock.TextAlignmentProperty, TextAlignment.Left));
                leftStyle.Setters.Add(new Setter(TextBlock.VerticalAlignmentProperty, System.Windows.VerticalAlignment.Center));
                leftStyle.Setters.Add(new Setter(TextBlock.PaddingProperty, new Thickness(10, 0, 0, 0)));

                grid.Columns.Add(new DataGridTextColumn { Header = "Coin", Binding = new Binding("Symbol"), Width = new DataGridLength(1, DataGridLengthUnitType.Star), FontWeight = FontWeights.Bold, ElementStyle = leftStyle });
                grid.Columns.Add(new DataGridTextColumn { Header = "Fiyat ($)", Binding = new Binding("Price"), Width = new DataGridLength(1, DataGridLengthUnitType.Auto), ElementStyle = leftStyle });

                var signalColumn = new DataGridTextColumn { Header = "İşlem", Binding = new Binding("SignalType"), Width = new DataGridLength(0.7, DataGridLengthUnitType.Star), FontWeight = FontWeights.Bold, ElementStyle = leftStyle };
                Style cellStyle = new Style(typeof(DataGridCell));
                cellStyle.Setters.Add(new Setter(DataGridCell.ForegroundProperty, new Binding("SignalColor")));

                var trigger = new Trigger { Property = DataGridCell.IsSelectedProperty, Value = true };
                trigger.Setters.Add(new Setter(DataGridCell.BackgroundProperty, (SolidColorBrush)new BrushConverter().ConvertFrom("#446A6A")));
                trigger.Setters.Add(new Setter(DataGridCell.BorderThicknessProperty, new Thickness(0)));
                cellStyle.Triggers.Add(trigger);
                signalColumn.CellStyle = cellStyle;
                grid.Columns.Add(signalColumn);

                if (i == 0)
                {
                    var riskColumn = new DataGridTextColumn { Header = "Risk", Binding = new Binding("RiskRatio"), Width = new DataGridLength(0.6, DataGridLengthUnitType.Star), FontWeight = FontWeights.Bold, ElementStyle = leftStyle };
                    Style riskStyle = new Style(typeof(DataGridCell));
                    riskStyle.Setters.Add(new Setter(DataGridCell.ForegroundProperty, new Binding("RiskColor")));
                    var riskTrigger = new Trigger { Property = DataGridCell.IsSelectedProperty, Value = true };
                    riskTrigger.Setters.Add(new Setter(DataGridCell.BackgroundProperty, (SolidColorBrush)new BrushConverter().ConvertFrom("#446A6A")));
                    riskTrigger.Setters.Add(new Setter(DataGridCell.BorderThicknessProperty, new Thickness(0)));
                    riskStyle.Triggers.Add(riskTrigger);
                    riskColumn.CellStyle = riskStyle;
                    grid.Columns.Add(riskColumn);
                }

                grid.Columns.Add(new DataGridTextColumn { Header = "Hedef ($)", Binding = new Binding("ExpectedPrice"), Width = new DataGridLength(1, DataGridLengthUnitType.Auto), ElementStyle = leftStyle });
                grid.Columns.Add(new DataGridTextColumn { Header = "AI Tahmini", Binding = new Binding("AiPercentage") { StringFormat = "P4" }, Width = new DataGridLength(1, DataGridLengthUnitType.Star), ElementStyle = leftStyle });
                grid.Columns.Add(new DataGridTextColumn { Header = "Hacim", Binding = new Binding("Volume") { StringFormat = "N0" }, Width = new DataGridLength(1, DataGridLengthUnitType.Star), ElementStyle = leftStyle });
            }
        }

        private async void RefreshListButton_Click(object sender, RoutedEventArgs e)
        {
            if (RefreshListButton == null) return;
            RefreshListButton.IsEnabled = false;
            AppendToTerminal("------------------------------------------------");
            AppendToTerminal("↻ Liste verileri yerel veritabanından güncelleniyor...");
            await RefreshMarketData(retryUntilSuccess: false);
            RefreshListButton.IsEnabled = true;
            ShowAlertDialog("Liste Yenilendi", "Veri seti güncellendi");
        }

        private async void StartScanButton_Click(object sender, RoutedEventArgs e)
        {
            if (StartScanButton == null) return;
            StartScanButton.IsEnabled = false;
            ScanProgressBar.Visibility = Visibility.Visible;
            ScanButtonText.Text = "TARANIYOR...";

            AppendToTerminal("------------------------------------------------");
            AppendToTerminal("🚀 PİYASA TARAMASI BAŞLATILIYOR (BINANCE)...");
            AppendToTerminal("⏳ Bu işlem 3-5 dakika sürebilir. Bekleyin.");

            try
            {
                await _marketService.TriggerScanAsync();
                AppendToTerminal("✅ Tarama tamamlandı. Sonuçlar listeye çekiliyor...");
                await RefreshMarketData(retryUntilSuccess: false);

                ShowAlertDialog("TARAMA TAMAMLANDI", "Piyasa taraması başarıyla tamamlandı ve veriler güncellendi.");
            }
            catch (Exception ex)
            {
                AppendToTerminal($"❌ TARAMA HATASI: {ex.Message}");
                ShowAlertDialog("HATA", ex.Message, isError: true);
            }
            finally
            {
                if (ScanProgressBar != null) ScanProgressBar.Visibility = Visibility.Collapsed;
                if (ScanButtonText != null) ScanButtonText.Text = "TARAMAYI BAŞLAT 🚀";
                StartScanButton.IsEnabled = true;
            }
        }

        private void SearchButton_Click(object sender, RoutedEventArgs e) => ApplyFilters();

        private async void MyDataGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (MyDataGrid.SelectedItem is CoinItem selectedCoin)
            {
                await LoadAndShowCoinChart(selectedCoin.Symbol);
            }
        }

        private void BackToListViewButton_Click(object sender, RoutedEventArgs e)
        {
            if (ListAndTerminalContainer != null) ListAndTerminalContainer.Visibility = Visibility.Visible;
            if (ChartContainer != null) ChartContainer.Visibility = Visibility.Collapsed;
        }

        private void MainWindow_Closing(object? sender, System.ComponentModel.CancelEventArgs e) => _backendService.KillProcess();

        private void AppendToTerminal(string? text)
        {
            if (string.IsNullOrEmpty(text)) return;
            Dispatcher.Invoke(() =>
            {
                if (TerminalOutput != null)
                {
                    TerminalOutput.AppendText($"> {text}\n");
                    TerminalOutput.ScrollToEnd();
                }
            });
        }

        // --- GRAFİK KODLARI ---
        private async Task LoadAndShowCoinChart(string symbol)
        {
            if (ListAndTerminalContainer != null) ListAndTerminalContainer.Visibility = Visibility.Collapsed;
            if (ChartContainer != null) ChartContainer.Visibility = Visibility.Visible;

            ScottPlot.Color bgColor = ScottPlot.Color.FromHex("#161A25");
            ScottPlot.Color gridColor = ScottPlot.Color.FromHex("#2B2F36");
            ScottPlot.Color axisColor = ScottPlot.Color.FromHex("#848E9C");

            WpfPlotPrice.Plot.Clear(); WpfPlotVolume.Plot.Clear();

            WpfPlotPrice.Plot.FigureBackground.Color = bgColor;
            WpfPlotPrice.Plot.DataBackground.Color = bgColor;
            WpfPlotPrice.Plot.Grid.MajorLineColor = gridColor;
            WpfPlotPrice.Plot.Axes.Color(axisColor);
            WpfPlotPrice.Plot.Axes.Left.IsVisible = false;
            WpfPlotPrice.Plot.Axes.Right.IsVisible = true;
            WpfPlotPrice.Plot.Axes.Right.TickLabelStyle.ForeColor = axisColor;
            WpfPlotPrice.Plot.Axes.Bottom.TickLabelStyle.IsVisible = false;

            WpfPlotVolume.Plot.FigureBackground.Color = bgColor;
            WpfPlotVolume.Plot.DataBackground.Color = bgColor;
            WpfPlotVolume.Plot.Grid.MajorLineColor = gridColor;
            WpfPlotVolume.Plot.Axes.Color(axisColor);
            WpfPlotVolume.Plot.Axes.Left.IsVisible = false;
            WpfPlotVolume.Plot.Axes.Right.IsVisible = true;
            WpfPlotVolume.Plot.Axes.Right.TickLabelStyle.ForeColor = axisColor;
            WpfPlotVolume.Plot.Axes.DateTimeTicksBottom();

            WpfPlotPrice.Plot.Title(""); WpfPlotVolume.Plot.Title("");
            WpfPlotPrice.Refresh(); WpfPlotVolume.Refresh();

            AppendToTerminal($"Grafik verisi çekiliyor: {symbol}...");
            string apiSymbol = symbol.Replace("/USDT", "");

            try
            {
                var response = await _chartHttpClient.GetStringAsync($"http://127.0.0.1:8000/get_coin_data/{apiSymbol}");
                using var jsonDoc = JsonDocument.Parse(response);

                if (jsonDoc.RootElement.ValueKind != JsonValueKind.Object) { WpfPlotPrice.Plot.Title("HATA: Veri Yok"); return; }

                List<OHLC> ohlcList = new();
                List<double> volumeList = new();

                if (jsonDoc.RootElement.TryGetProperty("last_indicators", out var indicatorsElement) && indicatorsElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var indicator in indicatorsElement.EnumerateArray().Reverse())
                    {
                        if (indicator.TryGetProperty("Open", out var o) && indicator.TryGetProperty("High", out var h) &&
                            indicator.TryGetProperty("Low", out var l) && indicator.TryGetProperty("Close", out var c) &&
                            indicator.TryGetProperty("Date", out var d) && DateTime.TryParse(d.GetString(), out DateTime dt))
                        {
                            ohlcList.Add(new OHLC(o.GetDouble(), h.GetDouble(), l.GetDouble(), c.GetDouble(), dt, TimeSpan.FromMinutes(15)));
                            double vol = 0;
                            if (indicator.TryGetProperty("Volume", out var v)) vol = v.GetDouble();
                            volumeList.Add(vol);
                        }
                    }
                }

                if (ohlcList.Count > 0)
                {
                    var candles = WpfPlotPrice.Plot.Add.Candlestick(ohlcList);
                    candles.RisingColor = ScottPlot.Color.FromHex("#24E500");
                    candles.FallingColor = ScottPlot.Color.FromHex("#ff0000");
                    candles.Axes.YAxis = WpfPlotPrice.Plot.Axes.Right;

                    double[] xs = ohlcList.Select(x => x.DateTime.ToOADate()).ToArray();
                    var ma7 = FinanceIndicators.CalculateSMA(ohlcList, 7);
                    var ma25 = FinanceIndicators.CalculateSMA(ohlcList, 25);
                    var ma99 = FinanceIndicators.CalculateSMA(ohlcList, 99);

                    AddLineToChart(WpfPlotPrice, xs, ma7, "#F0B90B");
                    AddLineToChart(WpfPlotPrice, xs, ma25, "#E0294A");
                    AddLineToChart(WpfPlotPrice, xs, ma99, "#8F519B");

                    var bars = new List<ScottPlot.Bar>();
                    for (int i = 0; i < ohlcList.Count; i++)
                    {
                        var color = ohlcList[i].Close >= ohlcList[i].Open ? ScottPlot.Color.FromHex("#0ECB8180") : ScottPlot.Color.FromHex("#F6465D80");
                        bars.Add(new ScottPlot.Bar { Position = ohlcList[i].DateTime.ToOADate(), Value = volumeList[i], ValueBase = 0, FillColor = color, Size = 0.007 });
                    }
                    var barPlot = WpfPlotVolume.Plot.Add.Bars(bars);
                    barPlot.Axes.YAxis = WpfPlotVolume.Plot.Axes.Right;

                    WpfPlotPrice.Plot.Axes.AutoScale();
                    WpfPlotVolume.Plot.Axes.AutoScale();
                    WpfPlotPrice.Plot.Axes.Link(WpfPlotVolume);

                    UpdateChartInfo(ohlcList.Last(), ma7.Last(), ma25.Last(), ma99.Last());
                    WpfPlotPrice.Refresh(); WpfPlotVolume.Refresh();
                    AppendToTerminal($"Grafik {symbol} için hazırlandı.");
                }
            }
            catch (Exception ex) { AppendToTerminal($"GRAFİK HATA: {ex.Message}"); }
        }

        private void AddLineToChart(IPlotControl plot, double[] xs, double[] ys, string hexColor)
        {
            var line = plot.Plot.Add.ScatterLine(xs, ys);
            line.Color = ScottPlot.Color.FromHex(hexColor);
            line.LineWidth = 1.5f;
            line.Axes.YAxis = plot.Plot.Axes.Right;
        }

        private void UpdateChartInfo(OHLC candle, double ma7, double ma25, double ma99)
        {
            Dispatcher.Invoke(() =>
            {
                if (ChartHeaderInfo != null) ChartHeaderInfo.Text = $"O: {candle.Open:0.00} H: {candle.High:0.00} L: {candle.Low:0.00} C: {candle.Close:0.00}";
                if (ChartIndicatorsInfo != null) ChartIndicatorsInfo.Text = $"MA(7): {ma7:0.00} MA(25): {ma25:0.00} MA(99): {ma99:0.00}";
            });
        }

        private void BtnMinimize_Click(object sender, RoutedEventArgs e) => this.WindowState = WindowState.Minimized;
        private void ExitFullScreen_Click(object sender, RoutedEventArgs e) => this.Close();
        private void Border_MouseEnter1(object sender, MouseEventArgs e) => SetBorderColor(sender, "#446A6A", Color.FromArgb(255, 100, 255, 255));
        private void Border_MouseLeave1(object sender, MouseEventArgs e) => SetBorderColor(sender, "Transparent", Color.FromArgb(255, 20, 71, 72));

        private void SetBorderColor(object sender, string hexBg, Color textColor)
        {
            if (sender is Border border)
            {
                border.Background = hexBg == "Transparent" ? Brushes.Transparent : new SolidColorBrush((Color)ColorConverter.ConvertFromString(hexBg));
                if (FindVisualChild<StackPanel>(border) is { } stackPanel && FindVisualChild<TextBlock>(stackPanel) is { } textBlock)
                    textBlock.Foreground = new SolidColorBrush(textColor);
            }
        }

        private void Border_MouseEnter(object sender, MouseEventArgs e) => AnimateSidebar(2, 8, 1);
        private void Border_MouseLeave(object sender, MouseEventArgs e) => AnimateSidebar(0.8, 9.2, 0);

        private void AnimateSidebar(double left, double right, double opacity)
        {
            AnimateGridLength(Solpanel, Solpanel.Width.Value, left, 0.3);
            AnimateGridLength(Sagpanel, Sagpanel.Width.Value, right, 0.3);
            var anim = new DoubleAnimation(opacity, TimeSpan.FromSeconds(0.3));

            if (HomePAGEText != null) HomePAGEText.BeginAnimation(UIElement.OpacityProperty, anim);
        }

        private static void AnimateGridLength(ColumnDefinition column, double from, double to, double durationSeconds)
        {
            var anim = new GridLengthAnimation { From = new GridLength(from, GridUnitType.Star), To = new GridLength(to, GridUnitType.Star), Duration = TimeSpan.FromSeconds(durationSeconds) };
            column.BeginAnimation(ColumnDefinition.WidthProperty, anim);
        }

        public static T? FindVisualChild<T>(DependencyObject parent) where T : DependencyObject
        {
            if (parent is null) return null;
            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(parent); i++)
            {
                if (VisualTreeHelper.GetChild(parent, i) is T tChild) return tChild;
                if (FindVisualChild<T>(VisualTreeHelper.GetChild(parent, i)) is { } childOfChild) return childOfChild;
            }
            return null;
        }
    }
    
}