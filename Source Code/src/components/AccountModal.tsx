// src/components/AccountModal.tsx
"use client";
import { X } from "@/assets/icons";
import Input from "./Input";
import { useEffect, useState } from "react";
import { useUser } from "@/hook/useUser";
import { showNotifyToast } from "./NotificationToast";
import { ApiRoute } from "@/app/api";
import { useRouter } from "next/navigation";

interface ModalProps {
	setIsOpen: (open: boolean) => void;
	closeOnClick?: boolean;
}

const AccountModal: React.FC<ModalProps> = ({
	setIsOpen,
	closeOnClick = true,
}) => {
	const { userId } = useUser();
	const [account, setAccount] = useState("");
	const [server, setServer] = useState("");
	const [password, setPassword] = useState(""); // Actual password value
	const [displayPassword, setDisplayPassword] = useState(""); // Password for display
	const [remember, setRemember] = useState(false);
	const [loading, setLoading] = useState(false);
	const router = useRouter();

	// Fetch stored password on mount with delay for navigation
	useEffect(() => {
		// Add delay before checking userId
		const timer = setTimeout(() => {
			if (userId == null) {
				router.push("/auth/login");
				return;
			}

			setLoading(true); // Start loading
			fetch(ApiRoute.getAccount(userId))
				.then((res) => res.json())
				.then((data) => {
					if (data.loginId) {
						setAccount(data.loginId);
						setPassword(data.password); // hashed password
						setDisplayPassword(data.remember ? "••••••••••" : ""); // Mask if encrypted
						setServer(data.server);
						setRemember(data.remember);
					}
				})
				.catch((err) => console.error(err))
				.finally(() => setLoading(false)); // Stop loading
		}, 1000); // 1000ms = 1 second delay

		// Cleanup timeout on unmount or if userId changes
		return () => clearTimeout(timer);
	}, [userId, router]);

	const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		setLoading(true); // Start loading

		try {
			const res = await fetch(ApiRoute.postAccount, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					userId,
					loginId: account,
					server: server.trim(),
					password: password.trim(),
					remember,
				}),
			});

			const data = await res.json();
			if (!res.ok) {
				showNotifyToast(
					data.message || "Failed to login broker account",
					"error"
				);
			} else {
				showNotifyToast(data.message, "success");

				// Update sessionStorage
				sessionStorage.setItem("user-account", account);
				sessionStorage.setItem("account-server", server);
				sessionStorage.setItem("account-password", data.encryptedPassword);
				sessionStorage.setItem("account-name", data.accountInfo.name);

				// Dispatch custom event to notify same-tab listeners
				window.dispatchEvent(new Event("sessionStorageUpdated"));

				// console.log(
				// 	"AccountModal: ",
				// 	account,
				// 	server,
				// 	data.encryptedPassword,
				// 	data.accountInfo.name
				// );

				setIsOpen(false);
			}
		} catch (error) {
			showNotifyToast("An unexpected error occurred", "error");
			console.error("Login error:", error);
		} finally {
			setLoading(false); // Stop loading
		}
	};

	const isStored = !!sessionStorage.getItem("user-account");

	return (
		<>
			{/* Modal */}
			<div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
				<div className="relative p-4 w-full max-w-md">
					{/* Modal Content */}
					<div className="relative bg-white rounded-lg shadow dark:bg-gray-700">
						{/* Modal Header */}
						<div className="flex items-center justify-between p-4 md:p-5 border-b rounded-t dark:border-gray-600 border-gray-200">
							<h3 className="text-xl font-semibold text-gray-900 dark:text-white">
								Login Broker Account
							</h3>
							{closeOnClick && (
								<button
									onClick={() => setIsOpen(false)}
									className="text-gray-400 bg-transparent hover:bg-gray-200 hover:text-gray-900 rounded-lg text-sm w-8 h-8 flex justify-center items-center dark:hover:bg-gray-600 dark:hover:text-white"
									disabled={loading} // Disable close button during loading
								>
									<X className="w-5 h-5" />
								</button>
							)}
						</div>

						{/* Modal Body */}
						<div className="p-4">
							<form className="space-y-4" onSubmit={handleLogin}>
								<Input
									id={"account"}
									label={"Account"}
									placeholder={"0123456789"}
									value={account}
									type={"number"}
									inputMode="numeric"
									onChange={(e) => setAccount(e.target.value)}
									required
									disabled={isStored || loading} // Disable during loading
								/>
								<Input
									id={"password"}
									label={"Password"}
									placeholder={"••••••••"}
									value={displayPassword}
									type={"password"}
									onChange={(e) => {
										setDisplayPassword(e.target.value); // Update display
										setPassword(e.target.value); // Update actual password
									}}
									required
									disabled={loading} // Disable during loading
								/>
								<Input
									id={"server"}
									label={"Server"}
									placeholder={"Broker-Server"}
									value={server}
									onChange={(e) => setServer(e.target.value)}
									required
									disabled={isStored || loading} // Disable during loading
								/>
								<div className="flex items-start">
									<div className="flex items-center h-5">
										<input
											id="remember"
											type="checkbox"
											checked={remember}
											className="w-4 h-4 border border-gray-300 rounded-sm bg-gray-50 focus:ring-3 focus:ring-blue-300 dark:bg-gray-600 dark:border-gray-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 dark:focus:ring-offset-gray-800"
											onChange={(e) => setRemember(e.target.checked)}
											disabled={loading} // Disable during loading
										/>
									</div>
									<label
										htmlFor="remember"
										className="ms-2 text-sm font-medium text-gray-900 dark:text-gray-300"
									>
										Save password
									</label>
								</div>
								<button
									type="submit"
									className={`w-full text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800 ${
										loading ? "opacity-50 cursor-not-allowed" : ""
									}`}
									disabled={loading} // Disable button during loading
								>
									{loading ? "Logging in..." : "Login to your account"}
								</button>
							</form>
						</div>
					</div>
				</div>
			</div>
		</>
	);
};

export default AccountModal;
