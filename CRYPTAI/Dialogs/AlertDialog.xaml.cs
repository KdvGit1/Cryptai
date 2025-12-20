using System.Windows;
using System.Windows.Media;
using Brushes = System.Windows.Media.Brushes;

namespace CRYPTAI
{
    public partial class AlertDialog : Window
    {
        public AlertDialog(string title, string message, bool isError = false)
        {
            InitializeComponent();
            TxtTitle.Text = title;
            TxtMessage.Text = message;

            if (isError)
            {
                TxtIcon.Text = "❌";
                TxtTitle.Foreground = Brushes.Red;
                // Border rengini de kodla değiştirebilirsin ama şimdilik basit kalsın
            }
        }

        private void BtnOk_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}