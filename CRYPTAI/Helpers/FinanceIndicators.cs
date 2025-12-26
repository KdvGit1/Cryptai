using System.Collections.Generic;
using ScottPlot; // ScottPlot OHLC yapısını kullanıyorsan

namespace CRYPTAI.Helpers
{
    public static class FinanceIndicators
    {
        public static double[] CalculateSMA(List<OHLC> candles, int period)
        {
            double[] sma = new double[candles.Count];
            for (int i = 0; i < candles.Count; i++)
            {
                if (i < period - 1)
                {
                    sma[i] = double.NaN;
                }
                else
                {
                    double sum = 0;
                    for (int j = 0; j < period; j++)
                    {
                        sum += candles[i - j].Close;
                    }
                    sma[i] = sum / period;
                }
            }
            return sma;
        }
    }
}