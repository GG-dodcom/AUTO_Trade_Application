// kept all the data queries
import { ApiRoute } from "@/app/api";

// Function to fetch MT5 account info
export async function getAccountInfo(
	userId: string,
	loginId: string,
	password: string,
	server: string
): Promise<any> {
	const accountInfoResponse = await fetch(
		ApiRoute.getMT5AccountInfo(userId, loginId, password, server),
		{
			method: "GET",
			headers: { "Content-Type": "application/json" },
		}
	);
	const accountInfoData = await accountInfoResponse.json();
	return accountInfoResponse.ok && accountInfoData.account_info
		? accountInfoData.account_info
		: null;
}

export const getAccountId = async (
	userId: number,
	loginId: number,
	server: string
) => {
	const storedAccountId = sessionStorage.getItem("account-id");
	if (storedAccountId) return storedAccountId;

	const url = new URL(ApiRoute.getAccountId, window.location.origin);
	url.searchParams.append("userId", userId.toString());
	url.searchParams.append("loginId", loginId.toString());
	url.searchParams.append("server", server);

	const response = await fetch(url.toString(), { credentials: "include" });
	if (!response.ok) {
		const errorData = await response.json();
		throw new Error(errorData.error || "Failed to fetch account ID");
	}

	const data = await response.json();
	sessionStorage.setItem("account-id", data.accountId.toString());
	return data.accountId;
};
