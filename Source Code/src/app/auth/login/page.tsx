"use client";
import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ApiRoute } from "@/app/api";
import Input from "@/components/Input";
import { showNotifyToast } from "@/components/NotificationToast";

export default function Login() {
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");
	const [isLoading, setIsLoading] = useState(false);
	const router = useRouter();

	useEffect(() => {
		console.log("Before clear:", localStorage.length, sessionStorage.length);
		localStorage.clear();
		sessionStorage.clear();
		console.log("After clear:", localStorage.length, sessionStorage.length);
	}, []);

	const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (!email || !password) {
			showNotifyToast("Please enter both email and password", "error");
			return;
		}

		setIsLoading(true);

		// Handle form submission, e.g., authenticate user
		try {
			const res = await fetch(ApiRoute.login, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ email, password }),
				credentials: "include", // Ensure cookies are sent
			});
			const response = await res.json();

			if (res.ok) {
				showNotifyToast(response.message, "success", "bottom-center", 1000);

				// Prefetch the home page immediately (Next.js will cache it)
				router.prefetch("/");

				// Store user data in sessionStorage (Not persistent)
				sessionStorage.setItem("user-id", response.user.id);
				sessionStorage.setItem("user-email", response.user.email);

				setTimeout(() => {
					router.push("/");
				}, 2000); // Wait for 2 seconds before routing
			} else {
				showNotifyToast(response.message || "An error occurred", "error");
			}
		} catch (error) {
			showNotifyToast("Network error occurred", "error");
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<div className="flex items-center justify-center !h-[100vh] bg-gray-100">
			<div className="w-full max-w-md p-8 space-y-6 bg-white rounded-lg shadow">
				<h2 className="text-2xl font-bold text-left text-gray-900">Login</h2>
				<form className="mt-8 space-y-6" onSubmit={handleLogin}>
					<div className="space-y-4">
						<Input
							id="email"
							label="Your Email"
							placeholder="name@example.com"
							type="email"
							value={email}
							onChange={(e) => setEmail(e.target.value)}
						/>
						<Input
							id="password"
							label="Password"
							placeholder="**********"
							type="password"
							value={password}
							onChange={(e) => setPassword(e.target.value)}
						/>
					</div>
					<button
						type="submit"
						disabled={isLoading}
						className={`flex justify-center w-full px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
							isLoading ? "opacity-50 cursor-not-allowed" : ""
						}`}
					>
						{isLoading ? "Logging in..." : "Login"}
					</button>
				</form>
				{/* Add signup link below the form */}
				<p className="mt-2 text-center text-sm text-gray-600">
					Don’t have an account?{" "}
					<Link
						href="/auth/signup"
						className="font-medium text-blue-600 hover:text-blue-500"
					>
						Sign up
					</Link>
				</p>
			</div>
		</div>
	);
}
