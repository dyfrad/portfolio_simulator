import type { Metadata } from "next";
import "./globals.css";

export const viewport = {
  width: "device-width",
  initialScale: 1,
};

export const metadata: Metadata = {
  title: "Portfolio Simulator - Advanced Monte Carlo Analysis",
  description: "Run sophisticated Monte Carlo simulations to understand your portfolio's potential performance. Optimize asset allocation, test different scenarios, and make data-driven investment decisions.",
  keywords: ["portfolio", "simulation", "monte carlo", "investment", "optimization", "financial planning"],
  authors: [{ name: "Portfolio Simulator Team" }],
  robots: "index, follow",
  openGraph: {
    title: "Portfolio Simulator - Advanced Monte Carlo Analysis",
    description: "Run sophisticated Monte Carlo simulations to understand your portfolio's potential performance.",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "Portfolio Simulator - Advanced Monte Carlo Analysis",
    description: "Run sophisticated Monte Carlo simulations to understand your portfolio's potential performance.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
