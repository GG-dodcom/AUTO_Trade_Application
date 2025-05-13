// src/components/Sidebar.tsx
import React, { useState } from "react";
import { usePathname } from "next/navigation";
import { DropdownMenu } from "./DropdownMenu";
import {
	Logo,
	Moon,
	Bell,
	User,
	Bar,
	ChevronDownIcon24Outline,
	ChevronUpIcon24Outline,
	AdjustmentsVertical,
	Cog,
	Globe,
} from "@/assets/icons";
import IconButton from "./IconButton";
import { ProfilePopup } from "./ProfilePopup";
import { SidebarData } from "@/data/sidebarData";
import Link from "next/link";
import Badge from "./Badge";
import RunModel from "./RunModel";
import { useUser } from "@/hook/useUser";

interface SidebarProps {
	onSidebarToggle: (isVisible: boolean) => void; // This is the callback to notify the parent about the sidebar visibility
	onRunModalToggle?: (isRunning: boolean) => void;
}

export interface SidebarItem {
	title: string;
	icon: any;
	link: string;
	hasDropdown?: boolean;
	subMenu?: { title: string; link: string }[];
	isFooterItem?: boolean;
	badge?: string;
}

const Sidebar: React.FC<SidebarProps> = ({
	onSidebarToggle,
	onRunModalToggle,
}) => {
	const currentPath = usePathname(); // Get the current path
	const [activeItem, setActiveItem] = useState<number | null>(null);
	const [sidebarOpen, setSidebarOpen] = useState(true);
	const { account, accName } = useUser();

	// Separate main and footer items
	const mainItems = SidebarData.filter((item) => !item.isFooterItem);
	const footerItems = SidebarData.filter((item) => item.isFooterItem);

	const toggleSidebar = () => {
		const newSidebarState = !sidebarOpen;
		setSidebarOpen(newSidebarState); // Toggle sidebar visibility
		onSidebarToggle(newSidebarState); // Notify the parent component about the new sidebar visibility state
	};

	return (
		<>
			{/* Navbar */}
			<nav
				className={`fixed top-0 z-50 w-full pt-2.5 pr-5 pb-2.5 pl-5 
          flex items-center justify-between h-14
          bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700
        `}
			>
				<div className="flex items-center gap-2">
					<div className="relative w-8 h-8">
						<Link href="/">
							<Logo />
						</Link>
					</div>
					<h1>AUTO Trade</h1>
					{/* IconButton to toggle Sidebar */}
					<IconButton
						Icon={Bar}
						iconClassName="w-6 h-6 text-gray-600 hover:text-gray-800 dark:text-gray-300 dark:hover:text-white"
						onClick={toggleSidebar}
					/>
				</div>
				<div className="flex items-center">
					{/* Dark Mode */}
					{/* <IconButton
						Icon={Moon}
						iconClassName="w-6 h-6"
						onClick={() => alert("Moon Clicked!")}
					/> */}

					{/* Notification */}
					{/* <IconButton
						Icon={Bell}
						iconClassName="w-6 h-6"
						onClick={() => alert("Bell Clicked!")}
					/> */}

					<div className="flex items-center gap-2">
						{/* Profile */}
						<DropdownMenu
							content={<ProfilePopup />}
							header={`${account} : ${accName}`}
							dropIcon={false}
							theme="hover:bg-gray-100 !focus:outline-none !focus:ring-0 !focus:ring-offset-0"
							leftIcon={
								<div>
									<User className="w-6 h-6" />
								</div>
							}
						/>
					</div>
				</div>
			</nav>

			{/* Sidebar */}
			<aside
				className={`fixed top-0 left-0 z-40 w-64 h-screen pt-[3.4rem] transition-transform duration-300 ease-in-out
        ${
					sidebarOpen
						? "transform translate-x-0"
						: "transform -translate-x-full"
				} bg-white border-r border-gray-200 dark:bg-gray-800 dark:border-gray-700`}
				aria-label="Sidebar"
			>
				{/* RunModel Component */}
				<div className="">
					<RunModel initialModel="XAUUSD Model" onToggle={onRunModalToggle} />
				</div>

				<div className="h-full pt-2 px-3 pb-4 overflow-y-auto bg-white dark:bg-gray-800">
					<ul className="space-y-2 font-medium">
						{/* Sidebar items */}
						{mainItems.map((item: SidebarItem, index) => (
							<div key={index}>
								<Link
									href={item.link}
									className={`w-full flex items-center gap-3 w-full p-2 rounded-lg text-gray-900 hover:bg-gray-100 transition dark:hover:bg-gray-700 group${
										currentPath === item.link ? "bg-gray-100" : ""
									}`}
									onClick={() =>
										setActiveItem(index === activeItem ? null : index)
									}
								>
									{/* Render the SVG */}
									<item.icon
										className={`w-6 h-6 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white ${
											currentPath === item.link
												? "text-gray-800"
												: "text-gray-400"
										}`}
									/>
									<span className="text-gray-900 font-medium flex-1 text-left">
										{item.title}
									</span>

									{/* Render Badge if item has notifications */}
									{item.badge && (
										<Badge text={item.badge} theme={"Red"} size={"Small"} />
									)}

									{item.subMenu && (
										<span className="right-2">
											{activeItem == index ? (
												<ChevronUpIcon24Outline className="w-4 h-4" />
											) : (
												<ChevronDownIcon24Outline className="w-4 h-4" />
											)}
										</span>
									)}
								</Link>
								{item.subMenu && activeItem === index && (
									<div className="pt-2 pl-11 space-y-2">
										{item.subMenu.map((subItem, subIndex) => (
											<Link
												href={subItem.link}
												key={subIndex}
												className="block text-gray-700 hover:text-gray-900"
											>
												{subItem.title}
											</Link>
										))}
									</div>
								)}
							</div>
						))}
						<div className="border-t my-4"></div>

						<nav className="space-y-2">
							{footerItems.map((item, index) => (
								<Link
									href={item.link}
									key={index}
									className={`flex items-center gap-3 w-full p-2 rounded-lg text-gray-900 hover:bg-gray-100 transition dark:hover:bg-gray-700 group${
										currentPath === item.link ? "bg-gray-100" : ""
									}`}
								>
									<item.icon
										className={`w-6 h-6 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white ${
											currentPath === item.link
												? "text-gray-800"
												: "text-gray-400"
										}`}
									/>
									<span className="text-gray-900 font-medium">
										{item.title}
									</span>
								</Link>
							))}
						</nav>

						{/* <div className="flex justify-center gap-4 p-4 mt-auto text-gray-500">
							<AdjustmentsVertical className="w-6 h-6" />
							<Globe className="w-6 h-6" />
							<Cog className="w-6 h-6" />
						</div> */}
					</ul>
				</div>
			</aside>
		</>
	);
};

export default Sidebar;
