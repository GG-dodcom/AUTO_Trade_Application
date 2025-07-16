// src/components/ClientLayout.tsx
"use client";
import { useState, useEffect } from "react";
import { usePathname } from "next/navigation"; // Import usePathname
import Sidebar from "@/components/Sidebar";
import { useUser } from "@/hook/useUser";
import AccountModal from "./AccountModal";

interface ClientLayoutProps {
	children: React.ReactNode;
}

export default function ClientLayout({ children }: ClientLayoutProps) {
	const pathname = usePathname(); // Get current route
	const [isSidebarVisible, setIsSidebarVisible] = useState(true);
	const [isAccModel, setIsAccModel] = useState(false);
	const { account, accName, accPassword, accServer } = useUser();

	const handleSidebarToggle = (isVisible: boolean) => {
		setIsSidebarVisible(isVisible);
	};

	useEffect(() => {
		// Add a delay before checking the values
		const timer = setTimeout(() => {
			// console.log(accName, accPassword, accServer);
			if (!account && !accName && !accPassword && !accServer) {
				setIsAccModel(true);
			} else {
				setIsAccModel(false);
			}
		}, 100); // Delay of 100ms

		// Cleanup the timeout if the effect re-runs or component unmounts
		return () => clearTimeout(timer);
	}, [account, accName, accPassword, accServer]);

	// Check if the current route starts with "/auth"
	const isAuthRoute = pathname.startsWith("/auth");

	// If on an auth route, render only children without Sidebar or modal
	if (isAuthRoute) {
		return <>{children}</>;
	}

	// Otherwise, render the full layout with Sidebar and modal
	return (
		<div>
			<Sidebar onSidebarToggle={handleSidebarToggle} />
			<div
				className={`flex-1 transition-all duration-300 ease-in-out mt-14 ${
					isSidebarVisible ? "ml-64" : "ml-0"
				}`}
			>
				{children}
			</div>
			{isAccModel && (
				<div className="flex justify-center items-center min-h-screen">
					<AccountModal setIsOpen={setIsAccModel} closeOnClick={false} />
				</div>
			)}
		</div>
	);
}
