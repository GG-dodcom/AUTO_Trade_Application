import { ChevronLeft, ChevronRight } from "@/assets/icons";
import React from "react";

interface PaginationProps {
	currentPage: number;
	itemsPerPage: number; // Number of items per page
	totalItems: number; // Total number of items
	onPageChange: (page: number) => void;
	className?: string;
}

export const Pagination: React.FC<PaginationProps> = ({
	currentPage,
	itemsPerPage,
	totalItems,
	onPageChange,
	className,
}) => {
	const totalPages = Math.ceil(totalItems / itemsPerPage);

	const getPageNumbers = () => {
		const pages = [];
		const maxVisiblePages = 5; // Show up to 5 pages for simplicity

		if (totalPages <= maxVisiblePages) {
			for (let i = 1; i <= totalPages; i++) pages.push(i);
		} else {
			const leftBoundary = Math.max(2, currentPage - 1);
			const rightBoundary = Math.min(totalPages - 1, currentPage + 1);

			pages.push(1); // Always show first page
			if (leftBoundary > 2) pages.push("...");

			for (let i = leftBoundary; i <= rightBoundary; i++) {
				pages.push(i);
			}

			if (rightBoundary < totalPages - 1) pages.push("...");
			pages.push(totalPages); // Always show last page
		}
		return pages;
	};

	const startItem = totalItems === 0 ? 0 : (currentPage - 1) * itemsPerPage + 1;
	const endItem = Math.min(currentPage * itemsPerPage, totalItems);

	return (
		<nav
			className={`flex items-center justify-between gap-4 p-2 ${className}`}
			aria-label="Pagination"
		>
			{/* Showing X-Y of Z */}
			<span className="text-sm text-gray-600">
				Showing
				<span className="font-medium text-gray-800">
					{startItem}-{endItem}
				</span>
				of <span className="font-medium text-gray-800">{totalItems}</span>
			</span>

			{/* Page Navigation */}
			<ul className="inline-flex items-center gap-1 text-sm">
				{/* Previous Button */}
				<li>
					<button
						onClick={() => onPageChange(currentPage - 1)}
						disabled={currentPage === 1}
						className="p-2 rounded-full border border-gray-200 bg-white text-gray-600 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
						aria-label="Previous page"
					>
						{/* Previous */}
						<ChevronLeft className="w-5 h-5" />
					</button>
				</li>

				{/* Page Numbers */}
				{getPageNumbers().map((page, index) => (
					<li key={index}>
						{page === "..." ? (
							<span className="px-3 py-1 text-gray-400">...</span>
						) : (
							<button
								onClick={() => onPageChange(Number(page))}
								className={`px-3 py-1 rounded-md border border-gray-200 transition-colors duration-200 ${
									currentPage === page
										? "bg-blue-500 text-white border-blue-500"
										: "bg-white text-gray-700 hover:bg-gray-100"
								}`}
								aria-label={`Page ${page}`}
								aria-current={currentPage === page ? "page" : undefined}
							>
								{page}
							</button>
						)}
					</li>
				))}

				{/* Next Button */}
				<li>
					<button
						onClick={() => onPageChange(currentPage + 1)}
						disabled={currentPage === totalPages}
						className="p-2 rounded-full border border-gray-200 bg-white text-gray-600 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
						aria-label="Next page"
					>
						{/* Next */}
						<ChevronRight className="w-5 h-5" />
					</button>
				</li>
			</ul>
		</nav>
	);
};
