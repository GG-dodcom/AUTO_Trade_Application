import React from "react";

interface BlockItemProps {
	children: React.ReactNode;
	className?: string;
}

const BlockItem: React.FC<BlockItemProps> = ({ children, className = "" }) => {
	return (
		<div
			className={`flex-1 w-full sm:w-1/2 lg:w-1/3 xl:w-1/4 border rounded-lg shadow p-4 md:p-6 bg-white ${className}`}
		>
			{children}
		</div>
	);
};

export default BlockItem;
