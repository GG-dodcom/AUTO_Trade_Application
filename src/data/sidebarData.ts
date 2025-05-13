// src/data/sidebarData.ts
import {
	Calendar,
	ChartPie,
	Cog,
	File,
	FileChartBar,
	QuestionCircle,
	User,
} from "@/assets/icons";

export const SidebarData = [
	{
		title: "Dashboard",
		icon: ChartPie,
		link: "/",
	},
	{
		title: "Trading Journal",
		icon: FileChartBar,
		link: "/pages/trading_journal",
	},
	{
		title: "Monthly Report",
		icon: Calendar,
		link: "/pages/monthly_report",
	},
	{
		title: "Account",
		icon: User,
		link: "#",
		hasDropdown: true,
		subMenu: [
			{ title: "Login", link: "/auth/login" },
			{ title: "Log Out", link: "/auth/logout" },
		],
	},
	// {
	// 	title: "Messages",
	// 	icon: File,
	// 	link: "#",
	// 	badge: 99, // Show notification count
	// },
	// {
	// 	title: "Sales",
	// 	icon: File,
	// 	link: "#",
	// 	hasDropdown: true,
	// 	subMenu: [
	// 		{ title: "Product List", link: "/pages/docs" },
	// 		{ title: "Billing", link: "#" },
	// 		{ title: "Invoice", link: "#" },
	// 	],
	// },
	{
		title: "Backtest Report",
		icon: File,
		link: "/pages/backtest_report",
		isFooterItem: true, // Marks it as a footer item
	},
	{
		title: "Setting",
		icon: Cog,
		link: "/pages/settings",
		isFooterItem: true,
	},
	{
		title: "User Manual",
		icon: QuestionCircle,
		link: "/pages/user_manual",
		isFooterItem: true,
	},
];
