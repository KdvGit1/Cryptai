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
                TxtIcon.Foreground = Brushes.Red;
                border1.BorderBrush = Brushes.Red;
                okButton.Background= Brushes.Red;
            }
        }

        private void BtnOk_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}