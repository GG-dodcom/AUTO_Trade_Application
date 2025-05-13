import React from "react";

interface IconButtonProps {
	Icon: React.FC<React.SVGProps<SVGSVGElement>>; // Expect an SVG component
	text?: string; // Optional text next to the icon
	onClick?: () => void;
	className?: string; // Custom styles for the button
	iconClassName?: string; // Custom styles for the icon
	textClassName?: string; // Custom styles for the text
}

const IconButton: React.FC<IconButtonProps> = ({
	Icon,
	text,
	onClick,
	className = "",
	iconClassName = "",
	textClassName = "",
}) => {
	return (
		<button
			className={`hover:bg-gray-100 rounded-lg flex items-center gap-2 justify-center w-auto h-10 transition px-3 ${className}`}
			onClick={onClick}
		>
			<Icon className={`w-6 h-6 text-gray-700 ${iconClassName}`} />
			{text && (
				<span className={`text-gray-900 font-medium ${textClassName}`}>
					{text}
				</span>
			)}
		</button>
	);
};

export default IconButton;
