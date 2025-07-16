// src/components/ProfilePopup.tsx
"use client";
import { ApiRoute } from "@/app/api";
import { useUser } from "@/hook/useUser";

export const ProfilePopup: React.FC = () => {
	const { email, account, accServer, accName } = useUser();

	// // State to store account data
	// const [accounts, setAccounts] = useState<Account[]>([]);
	// const [loading, setLoading] = useState(true);
	// const [error, setError] = useState<string | null>(null);

	// // Fetch account data on mount
	// useEffect(() => {
	// 	async function fetchAccount() {
	// 		try {
	// 			const res = await fetch(`/api/account?userId=${userId}`);
	// 			if (!res.ok) throw new Error("Failed to fetch account data");
	// 			const data = await res.json();
	// 			setAccounts(data); // Set fetched account data
	// 		} catch (err) {
	// 			setError((err as Error).message);
	// 		} finally {
	// 			setLoading(false);
	// 		}
	// 	}
	// 	fetchAccount();
	// }, []);

	// // Function to switch account (Replace with actual logic)
	// function switchAccount(accountId: string, loginId: string, server: string) {
	// 	console.log(`Switching to account ${loginId} on ${server}`);
	// }

	// List of resources
	const resourcesTextLinks = [{ text: "Settings", link: "/pages/settings" }];

	// Logout handler
	const handleLogout = async () => {
		await fetch(ApiRoute.logout, { method: "POST" });
	};

	return (
		<div className="divide-y divide-gray-100 dark:divide-gray-600">
			{/* User Info */}
			<div className="px-4 py-3 text-sm text-gray-900 dark:text-white">
				<div>{accName}</div>
				<div className="font-medium truncate">{email}</div>
			</div>

			{/* Account List */}
			<div className="px-4 py-3 text-sm text-gray-700 dark:text-gray-200">
				<div className="text-gray-900">Broker Account</div>
				<div>{account}</div>
				<div>{accServer}</div>
				{/* {accounts.map((acc, index) => (
					<LinksItem
						key={index}
						onClick={() => switchAccount(acc.id, acc.loginId, acc.server)}
					>
						{acc.loginId} - {acc.server}
					</LinksItem>
				))} */}
			</div>

			{/* Resources Links */}
			<ul className="py-2 text-sm text-gray-700 dark:text-gray-200">
				{/* Resources Links */}
				{resourcesTextLinks.map((item, index) => (
					<LinksItem key={index} link={item.link}>
						{item.text}
					</LinksItem>
				))}
				{/* Logout */}
				<a
					href="/auth/login"
					className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
					onClick={handleLogout}
				>
					Logout
				</a>
			</ul>
		</div>
	);
};

// LinksItem component
interface LinksItemProps {
	link?: string;
	children: React.ReactNode;
	onClick?: () => void;
}

const LinksItem: React.FC<LinksItemProps> = ({ link, children, onClick }) => {
	return (
		<li>
			<a
				href={link}
				onClick={onClick}
				className="block px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-600 dark:hover:text-white"
			>
				{children}
			</a>
		</li>
	);
};
