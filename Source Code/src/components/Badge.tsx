import { X } from "@/assets/icons";
import React from "react";
import clsx from "clsx";

interface BadgeProps {
	type?: "Basic" | "Circle" | "Only Icon";
	text?: any;
	theme:
		| "Gray"
		| "Red"
		| "Yellow"
		| "Green"
		| "Blue"
		| "Indigo"
		| "Purple"
		| "Pink";
	size?: "Small" | "Large";
	icon?: React.ReactNode;
	remove_button?: boolean;
}

const themeColors: Record<BadgeProps["theme"], string> = {
	Gray: "bg-gray-100 text-gray-800",
	Red: "bg-red-100 text-red-800",
	Yellow: "bg-yellow-100 text-yellow-800",
	Green: "bg-green-100 text-green-800",
	Blue: "bg-blue-100 text-blue-800",
	Indigo: "bg-indigo-100 text-indigo-800",
	Purple: "bg-purple-100 text-purple-800",
	Pink: "bg-pink-100 text-pink-800",
};

const Badge: React.FC<BadgeProps> = ({
	text,
	type = "Basic",
	size = "Small",
	theme,
	icon,
	remove_button,
}) => {
	const sizeClasses =
		size === "Small" ? "text-xs px-2.5 py-0.5" : "text-sm px-3 py-0.5";
	const roundedClass =
		type === "Circle" ? "rounded-full w-5 h-5" : "rounded-md w-fit";

	return (
		<div
			className={clsx(
				"font-medium flex items-center justify-center gap-1",
				themeColors[theme],
				size === "Small" ? "text-xs px-2.5 py-0.5" : "text-sm px-3 py-0.5",
				type === "Circle" ? "rounded-full w-5 h-5" : "rounded-md w-fit"
			)}
		>
			{icon && <div className="w-3.5 h-3.5 text-gray-800">{icon}</div>}
			{text}
			{remove_button && <X className="w-3 h-3" />}
		</div>
	);
};

export default Badge;
