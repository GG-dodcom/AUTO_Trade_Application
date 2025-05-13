"use client"; // Mark as client-side component

import { useState, useEffect } from "react";

export default function BackTestReport() {
	const [htmlContent, setHtmlContent] = useState<string>("");
	const [isLoading, setIsLoading] = useState<boolean>(true);

	useEffect(() => {
		// Fetch the HTML file from the public folder
		fetch("/xauusd_trading_report.html")
			.then((res) => {
				if (!res.ok) throw new Error("Failed to fetch HTML file");
				return res.text();
			})
			.then((html) => {
				// Extract content inside <body> tags
				const bodyContent = html.replace(
					/.*<body[^>]*>(.*?)<\/body>.*/is,
					"$1"
				);
				setHtmlContent(bodyContent);
				setIsLoading(false);
			})
			.catch((err) => {
				console.error("Error fetching HTML:", err);
				setHtmlContent("<p>Error loading trading report</p>");
				setIsLoading(false);
			});
	}, []);

	if (isLoading) {
		return <div>Loading trading report...</div>;
	}

	return <div dangerouslySetInnerHTML={{ __html: htmlContent }} />;
}
