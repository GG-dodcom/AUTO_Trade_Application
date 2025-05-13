"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

const Logout = () => {
	const router = useRouter();

	// Handle the logout process
	const handleLogout = async () => {
		try {
			// Make a POST request to the logout API
			const response = await fetch("/api/logout", {
				method: "POST",
				credentials: "include", // Ensure cookies are sent with the request
			});

			if (!response.ok) {
				throw new Error("Logout API request failed");
			}

			// Clear all sessionStorage and localStorage data synchronously
			sessionStorage.clear();
			localStorage.clear();

			// Verify clearing (optional debug step)
			console.log("sessionStorage after clear:", sessionStorage.length); // Should be 0
			console.log("localStorage after clear:", localStorage.length); // Should be 0

			// Wait briefly to ensure clearing takes effect
			await new Promise((resolve) => setTimeout(resolve, 100));

			// Redirect to login page
			router.push("/auth/login");
		} catch (error) {
			console.error("Logout error:", error);
			// Ensure storage is cleared even if API fails
			sessionStorage.clear();
			localStorage.clear();
			router.push("/auth/login"); // Redirect anyway to ensure user is logged out
		}
	};

	// Use useEffect to trigger logout on component mount
	useEffect(() => {
		handleLogout();
		// No need to include router in dependencies since we only want this to run once on mount
	}, []);

	// Show a loading message while logging out
	return (
		<div className="flex items-center justify-center h-screen">
			<p className="text-gray-600">Logging out...</p>
		</div>
	);
};

export default Logout;
