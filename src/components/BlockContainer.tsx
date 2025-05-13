import React from "react";

interface BlockContainerProps {
	children: React.ReactNode;
	className?: string;
}

const BlockContainer: React.FC<BlockContainerProps> = ({
	children,
	className = "",
}) => {
	return (
		<div className={`relative flex flex-wrap gap-4 p-4 ${className}`}>
			{children}
		</div>
	);
};

export default BlockContainer;
