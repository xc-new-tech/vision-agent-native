"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import * as Tooltip from "@radix-ui/react-tooltip";

interface VersionSelectorProps {
  onVersionSelect: (version: 'v2' | 'v3') => void;
}

const versionInfo = {
  v3: {
    title: "VisionAgent",
    description: "单一循环架构，适合日常视觉任务。",
    features: [
      "支持图像输入",
      "工具调用与文件化结果",
      "输出为最终结论",
    ]
  },
  v2: {
    title: "VisionAgent",
    description: "与 v3 相同，仅保留兼容入口。",
    features: [
      "与 v3 行为一致",
      "保留 UI 版本选择",
    ]
  }
};

export function VersionSelector({ onVersionSelect }: VersionSelectorProps) {
  return (
    <Tooltip.Provider>
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <Card className="p-8 max-w-md w-full mx-4 bg-white">
          <div className="text-center space-y-6">
            <h2 className="text-2xl font-bold text-gray-900">
              Choose Version
            </h2>
            <p className="text-gray-600">
              Select which version of the chat API you would like to use:
            </p>
            <div className="space-y-3">
              <Tooltip.Root>
                <Tooltip.Trigger asChild>
                  <Button
                    onClick={() => onVersionSelect('v3')}
                    className="w-full py-3 text-lg border border-gray-300 bg-white text-gray-900 hover:bg-gray-50"
                    variant="outline"
                  >
                    VisionAgentV3
                  </Button>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content
                    className="bg-gray-900 text-white p-4 rounded-md shadow-lg max-w-xs z-50"
                    sideOffset={5}
                  >
                    <div className="space-y-2">
                      <h3 className="font-semibold">{versionInfo.v3.title}</h3>
                      <p className="text-sm text-gray-300">{versionInfo.v3.description}</p>
                      <ul className="text-sm space-y-1">
                        {versionInfo.v3.features.map((feature, index) => (
                          <li key={index} className="flex items-start">
                            <span className="mr-2">•</span>
                            <span>{feature}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <Tooltip.Arrow className="fill-gray-900" />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
              
              <Tooltip.Root>
                <Tooltip.Trigger asChild>
                  <Button
                    onClick={() => onVersionSelect('v2')}
                    className="w-full py-3 text-lg border border-gray-300 bg-white text-gray-900 hover:bg-gray-50"
                    variant="default"
                  >
                    VisionAgentV2
                  </Button>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content
                    className="bg-gray-900 text-white p-4 rounded-md shadow-lg max-w-xs z-50"
                    sideOffset={5}
                  >
                    <div className="space-y-2">
                      <h3 className="font-semibold">{versionInfo.v2.title}</h3>
                      <p className="text-sm text-gray-300">{versionInfo.v2.description}</p>
                      <ul className="text-sm space-y-1">
                        {versionInfo.v2.features.map((feature, index) => (
                          <li key={index} className="flex items-start">
                            <span className="mr-2">•</span>
                            <span>{feature}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <Tooltip.Arrow className="fill-gray-900" />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            </div>
          </div>
        </Card>
      </div>
    </Tooltip.Provider>
  );
}
