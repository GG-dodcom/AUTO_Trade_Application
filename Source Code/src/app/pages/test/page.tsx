"use client";
import { LineChart } from "@/components/LineChart";
import React from "react";

const Test: React.FC = () => {
	return (
		<div className="p-4">
			<LineChart
				height={350}
				title="Hello"
				seriesData={[10, 41, 35, 51, 49, 62, 69, 91, 148]}
				categories={[
					"Jan",
					"Feb",
					"Mar",
					"Apr",
					"May",
					"Jun",
					"Jul",
					"Aug",
					"Sep",
				]}
			/>
		</div>
	);
};

export default Test;
