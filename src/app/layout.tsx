import type { Metadata } from "next";

import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

export const metadata: Metadata = {
  title: "MRFSeg",
  description: "Conditional random field segmentation of CT images",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // Dark by default: this is a medical image viewer, and a light chrome around a
    // dark canvas makes window/level judgements harder.
    <html lang="en" className="dark font-sans">
      <body className="overflow-hidden antialiased">
        {children}
        <Toaster />
      </body>
    </html>
  );
}
