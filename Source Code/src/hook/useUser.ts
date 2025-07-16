import { getAccountId } from "@/lib/data";
import { useState, useEffect } from "react";

export const useUser = () => {
	const [userId, setUserId] = useState<number | null>(null);
	const [email, setEmail] = useState<string | null>(null);
	const [account, setAccount] = useState<number | null>(null);
	const [accountId, setAccountId] = useState<number | null>(null);
	const [accName, setAccName] = useState<string | null>(null);
	const [accPassword, setAccPassword] = useState<string | null>(null);
	const [accServer, setAccServer] = useState<string | null>(null);

	const updateUserData = () => {
		setUserId(Number(sessionStorage.getItem("user-id")) || null);
		setEmail(sessionStorage.getItem("user-email") || null);
		setAccount(Number(sessionStorage.getItem("user-account")) || null);
		setAccountId(Number(sessionStorage.getItem("account-id")) || null);
		setAccName(sessionStorage.getItem("account-name") || null);
		setAccPassword(sessionStorage.getItem("account-password") || null);
		setAccServer(sessionStorage.getItem("account-server") || null);
	};

	async function fetchAccountId() {
		if (accountId === null && userId && accName && accServer) {
			try {
				// Assuming accName is loginId (adjust if loginId is stored differently)
				const fetchedAccountId = await getAccountId(
					userId,
					Number(account),
					accServer
				);
				setAccountId(fetchedAccountId);
				// Update sessionStorage to keep it in sync
				sessionStorage.setItem("account-id", fetchedAccountId.toString());
			} catch (error) {
				console.error("Failed to fetch accountId:", error);
			}
		}
	}

	useEffect(() => {
		// Initial sync from sessionStorage
		updateUserData();
		fetchAccountId();

		// Listen for cross-tab storage events
		const handleStorageChange = (event: StorageEvent) => {
			if (event.storageArea === sessionStorage) {
				updateUserData();
			}
		};

		// Listen for same-tab custom event
		const handleCustomStorageUpdate = () => {
			updateUserData();
		};

		window.addEventListener("storage", handleStorageChange);
		window.addEventListener("sessionStorageUpdated", handleCustomStorageUpdate);

		return () => {
			window.removeEventListener("storage", handleStorageChange);
			window.removeEventListener(
				"sessionStorageUpdated",
				handleCustomStorageUpdate
			);
		};
	}, []);

	useEffect(() => {
		fetchAccountId();
	}, [userId, accName, accServer, accountId]);

	return {
		userId,
		email,
		account,
		accountId,
		accPassword,
		accServer,
		accName,
	};
};
