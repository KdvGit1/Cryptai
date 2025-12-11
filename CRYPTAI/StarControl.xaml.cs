using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Media.Animation; 
using UserControl = System.Windows.Controls.UserControl;

namespace CRYPTAI
{
    public partial class StarControl : UserControl
    {
        public StarControl()
        {
            InitializeComponent();
            UpdateImageImmediately();
        }
        public static readonly DependencyProperty IsStarredProperty =
            DependencyProperty.Register(
                nameof(IsStarred),
                typeof(bool),
                typeof(StarControl),
                new PropertyMetadata(false, OnIsStarredChanged));

        public bool IsStarred
        {
            get => (bool)GetValue(IsStarredProperty);
            set => SetValue(IsStarredProperty, value);
        }

        private static void OnIsStarredChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is StarControl control)
            {
                control.AnimateImageChange();
            }
        }
        private void AnimateImageChange()
        {
            if (PART_StarImage == null) return;

            var fadeOut = new DoubleAnimation
            {
                To = 0.0,
                Duration = TimeSpan.FromSeconds(0.05) 
            };

            fadeOut.Completed += (s, e) =>
            {
                UpdateImageImmediately();

                var fadeIn = new DoubleAnimation
                {
                    From = 0.0,
                    To = 1.0,
                    Duration = TimeSpan.FromSeconds(0.05)
                };
                PART_StarImage.BeginAnimation(OpacityProperty, fadeIn);
            };

            PART_StarImage.BeginAnimation(OpacityProperty, fadeOut);
        }

        private void UpdateImageImmediately()
        {
            if (PART_StarImage == null) return;
            var uri = IsStarred ? "/IMGS/LStar.png" : "/IMGS/Star.png";
            PART_StarImage.Source = new BitmapImage(new Uri(uri, UriKind.Relative));
        }

        private void PART_StarImage_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            IsStarred = !IsStarred; 
        }
    }
}