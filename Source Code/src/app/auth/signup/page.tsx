"use client";
import Input from "@/components/Input";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { ApiRoute } from "@/app/api";
import { showNotifyToast } from "@/components/NotificationToast";
import { ToastContainer } from "react-toastify";

export default function SignUp() {
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");
	const [confirm_password, setConfirm_password] = useState("");
	const router = useRouter();

	const handleSignUp = async (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();

		// Check if passwords match before sending the request
		if (password !== confirm_password) {
			showNotifyToast("Passwords do not match!", "error");
			return; // Stop execution if passwords don't match
		}

		// Send request to server
		const res = await fetch(ApiRoute.signup, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email, password }),
		});
		const response = await res.json();

		if (res.ok) {
			showNotifyToast(
				response.message ||
					"Account created successfully! Redirecting to login page...",
				"success",
				"bottom-center",
				1000
			);

			// Prefetch the login page immediately (Next.js will cache it)
			router.prefetch("/auth/login");

			setTimeout(() => {
				router.push("/auth/login");
			}, 2000); // Wait for 2 seconds before routing
		} else {
			showNotifyToast(response.message || "An error occurred", "error");
		}
	};

	return (
		<div className="flex items-center justify-center min-h-screen bg-gray-100">
			<div className="w-full max-w-md p-8 space-y-6 bg-white rounded-lg shadow">
				<h2 className="text-2xl font-bold text-left text-gray-900">Sign Up</h2>
				<form className="mt-8 space-y-6" onSubmit={handleSignUp}>
					<div className="space-y-4">
						<Input
							id={"email"}
							label={"Your Email"}
							placeholder={"name@example.com"}
							type={"email"}
							value={email}
							onChange={(e) => setEmail(e.target.value)}
						/>
						<Input
							id={"password"}
							label={"Password"}
							placeholder={"**********"}
							type={"password"}
							value={password}
							onChange={(e) => setPassword(e.target.value)}
						/>
						<Input
							id={"confirm_password"}
							label={"Confirm Password"}
							placeholder={"**********"}
							type={"password"}
							value={confirm_password}
							onChange={(e) => setConfirm_password(e.target.value)}
						/>
					</div>
					<button
						type="submit"
						className="flex justify-center w-full px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
					>
						Sign Up
					</button>
				</form>
			</div>
			<ToastContainer />
		</div>
	);
}
