#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Comprehensive RapidOCR Demo - Full Functionality Showcase
=========================================================

This demo showcases the complete functionality of RapidOCR including:
- Basic OCR (detection + recognition)
- Individual component testing (detection only, classification only, recognition only)
- Multiple language support
- Different model configurations
- Batch processing
- Visualization and output formatting
- Performance benchmarking
- Error handling and edge cases
- Interactive features

Author: Demo Enhancement for RapidOCR
Contact: Enhanced demo for full functionality showcase
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class MockRapidOCR:
    """
    Mock RapidOCR class for offline demonstration when models can't be downloaded.
    Provides realistic mock responses based on test images.
    """
    
    def __init__(self, **kwargs):
        self.mock_results = {
            "ch_en_num.jpg": {
                "boxes": [
                    [[75, 25], [295, 25], [295, 55], [75, 55]],
                    [[75, 65], [195, 65], [195, 85], [75, 85]],
                    [[75, 95], [245, 95], [245, 115], [75, 115]],
                ],
                "txts": ["正品促销", "Special Offer", "Quality Guaranteed"],
                "scores": [0.98, 0.95, 0.92]
            },
            "text_rec.jpg": {
                "boxes": [[[0, 0], [100, 0], [100, 30], [0, 30]]],
                "txts": ["韩国小馆"],
                "scores": [0.96]
            },
            "en.jpg": {
                "boxes": [[[0, 0], [50, 0], [50, 30], [0, 30]]],
                "txts": ["3"],
                "scores": [0.99]
            }
        }
        logging.info("[MOCK] Using mock RapidOCR for offline demonstration")
    
    def __call__(self, img_input, **kwargs):
        """Mock OCR processing"""
        if isinstance(img_input, str):
            if img_input.startswith('http'):
                filename = "ch_en_num.jpg"  # Default for URLs
            else:
                filename = Path(img_input).name
        else:
            filename = "ch_en_num.jpg"  # Default for other inputs
            
        mock_data = self.mock_results.get(filename, self.mock_results["ch_en_num.jpg"])
        
        # Simulate processing time
        time.sleep(0.1)
        
        return MockOCRResult(
            boxes=mock_data["boxes"],
            txts=mock_data["txts"],
            scores=mock_data["scores"],
            img=self._load_mock_image(img_input)
        )
    
    def _load_mock_image(self, img_input):
        """Load mock image or create a placeholder"""
        try:
            if isinstance(img_input, str) and not img_input.startswith('http'):
                return cv2.imread(img_input)
            else:
                # Create a mock image
                img = np.ones((100, 300, 3), dtype=np.uint8) * 255
                cv2.putText(img, "Mock Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                return img
        except:
            img = np.ones((100, 300, 3), dtype=np.uint8) * 255
            cv2.putText(img, "Mock Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            return img


class MockOCRResult:
    """Mock OCR result object with visualization capabilities"""
    
    def __init__(self, boxes=None, txts=None, scores=None, img=None, cls_res=None, word_results=None):
        self.boxes = boxes
        self.txts = txts
        self.scores = scores
        self.img = img
        self.cls_res = cls_res or []
        self.word_results = word_results or []
        
    def __len__(self):
        return len(self.txts) if self.txts else 0
    
    def to_json(self):
        """Convert result to JSON format"""
        if not self.txts:
            return []
        
        results = []
        for i, txt in enumerate(self.txts):
            result = {
                "txt": txt,
                "score": self.scores[i] if self.scores and i < len(self.scores) else 0.5
            }
            if self.boxes and i < len(self.boxes):
                result["box"] = self.boxes[i]
            results.append(result)
        return results
    
    def to_markdown(self):
        """Convert result to markdown table format"""
        if not self.txts:
            return "No text detected"
        
        markdown = "| Text | Confidence |\n|------|------------|\n"
        for i, txt in enumerate(self.txts):
            score = self.scores[i] if self.scores and i < len(self.scores) else 0.5
            markdown += f"| {txt} | {score:.2f} |\n"
        return markdown
    
    def vis(self, save_path=None):
        """Visualize OCR results"""
        if self.img is None:
            # Create a mock visualization
            img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.putText(img, "Mock Visualization", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            img = self.img.copy()
        
        # Draw bounding boxes and text
        if self.boxes and self.txts:
            for i, (box, txt) in enumerate(zip(self.boxes, self.txts)):
                if len(box) == 4:
                    pts = np.array(box, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                    
                    # Add text label
                    cv2.putText(img, txt, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"Visualization saved to: {save_path}")
        
        return img


class RapidOCRDemo:
    """Comprehensive demo class for RapidOCR functionality"""
    
    def __init__(self, use_mock=False):
        self.use_mock = use_mock
        self.engine = None
        self.test_images_dir = Path(__file__).parent / "python" / "tests" / "test_files"
        self.results_dir = Path("demo_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize OCR engine
        self._initialize_engine()
        
        # Performance tracking
        self.performance_stats = []
        
    def _initialize_engine(self):
        """Initialize RapidOCR engine or fallback to mock"""
        try:
            if not self.use_mock:
                from rapidocr import RapidOCR
                self.engine = RapidOCR()
                print("✓ RapidOCR engine initialized successfully")
            else:
                raise ImportError("Using mock by request")
        except Exception as e:
            print(f"⚠ Could not initialize RapidOCR engine: {e}")
            print("📝 Falling back to mock engine for demonstration")
            self.engine = MockRapidOCR()
            self.use_mock = True
    
    def run_basic_ocr_demo(self):
        """Demonstrate basic OCR functionality"""
        print("\n" + "="*60)
        print("🔍 BASIC OCR DEMONSTRATION")
        print("="*60)
        
        test_images = ["ch_en_num.jpg", "text_rec.jpg", "en.jpg"]
        
        for img_name in test_images:
            img_path = self.test_images_dir / img_name
            if not img_path.exists():
                print(f"⚠ Test image not found: {img_path}")
                continue
                
            print(f"\n📷 Processing: {img_name}")
            
            start_time = time.time()
            try:
                result = self.engine(str(img_path))
                processing_time = time.time() - start_time
                
                self.performance_stats.append({
                    "image": img_name,
                    "processing_time": processing_time,
                    "text_count": len(result) if result else 0
                })
                
                print(f"⏱ Processing time: {processing_time:.3f}s")
                print(f"📝 Detected {len(result)} text regions")
                
                if result and result.txts:
                    print("📋 Detected text:")
                    for i, (txt, score) in enumerate(zip(result.txts, result.scores or [])):
                        print(f"  {i+1}. {txt} (confidence: {score:.2f})")
                    
                    # Save visualization
                    vis_path = self.results_dir / f"{img_name}_basic_vis.png"
                    result.vis(str(vis_path))
                    
                    # Save JSON output
                    json_path = self.results_dir / f"{img_name}_result.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result.to_json(), f, ensure_ascii=False, indent=2)
                    
                    # Save markdown output
                    md_path = self.results_dir / f"{img_name}_result.md"
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(f"# OCR Results for {img_name}\n\n")
                        f.write(result.to_markdown())
                        
                else:
                    print("❌ No text detected")
                    
            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")
                if not self.use_mock:
                    traceback.print_exc()
    
    def run_component_testing(self):
        """Demonstrate individual component testing"""
        print("\n" + "="*60)
        print("🔧 COMPONENT TESTING DEMONSTRATION")
        print("="*60)
        
        if self.use_mock:
            print("📝 Component testing simulation (mock mode)")
            components = ["Detection Only", "Classification Only", "Recognition Only"]
            for component in components:
                print(f"✓ {component}: Simulated successful")
            return
        
        test_img = self.test_images_dir / "ch_en_num.jpg"
        if not test_img.exists():
            print("⚠ Test image not found for component testing")
            return
            
        print(f"📷 Testing components with: {test_img.name}")
        
        # Test detection only
        try:
            print("\n🎯 Testing Detection Only...")
            result = self.engine(str(test_img), use_det=True, use_cls=False, use_rec=False)
            print(f"✓ Detected {len(result)} regions")
            vis_path = self.results_dir / f"{test_img.name}_det_only.png"
            result.vis(str(vis_path))
        except Exception as e:
            print(f"❌ Detection test failed: {e}")
        
        # Test classification only
        try:
            print("\n📐 Testing Classification Only...")
            result = self.engine(str(test_img), use_det=False, use_cls=True, use_rec=False)
            print(f"✓ Classification completed")
        except Exception as e:
            print(f"❌ Classification test failed: {e}")
        
        # Test recognition only
        try:
            print("\n📝 Testing Recognition Only...")
            result = self.engine(str(test_img), use_det=False, use_cls=False, use_rec=True)
            print(f"✓ Recognition completed")
        except Exception as e:
            print(f"❌ Recognition test failed: {e}")
    
    def run_batch_processing_demo(self):
        """Demonstrate batch processing capabilities"""
        print("\n" + "="*60)
        print("📚 BATCH PROCESSING DEMONSTRATION")
        print("="*60)
        
        # Find all test images
        image_files = list(self.test_images_dir.glob("*.jpg")) + list(self.test_images_dir.glob("*.png"))
        image_files = [f for f in image_files if f.is_file()][:5]  # Limit to 5 images
        
        if not image_files:
            print("⚠ No test images found for batch processing")
            return
            
        print(f"🔄 Processing {len(image_files)} images in batch...")
        
        batch_results = []
        total_start_time = time.time()
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\n📷 [{i}/{len(image_files)}] Processing: {img_path.name}")
            
            start_time = time.time()
            try:
                result = self.engine(str(img_path))
                processing_time = time.time() - start_time
                
                batch_result = {
                    "filename": img_path.name,
                    "processing_time": processing_time,
                    "text_count": len(result) if result else 0,
                    "detected_text": result.txts if result and result.txts else [],
                    "confidence_scores": result.scores if result and result.scores else []
                }
                batch_results.append(batch_result)
                
                print(f"  ⏱ Time: {processing_time:.3f}s")
                print(f"  📝 Texts: {len(result) if result else 0}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                batch_results.append({
                    "filename": img_path.name,
                    "error": str(e)
                })
        
        total_time = time.time() - total_start_time
        
        # Save batch results
        batch_report_path = self.results_dir / "batch_processing_report.json"
        with open(batch_report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_processing_time": total_time,
                "average_time_per_image": total_time / len(image_files),
                "total_images": len(image_files),
                "successful_processes": len([r for r in batch_results if "error" not in r]),
                "results": batch_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"  📷 Total images: {len(image_files)}")
        print(f"  ⏱ Total time: {total_time:.3f}s")
        print(f"  📈 Average time per image: {total_time/len(image_files):.3f}s")
        print(f"  💾 Report saved: {batch_report_path}")
    
    def run_error_handling_demo(self):
        """Demonstrate error handling and edge cases"""
        print("\n" + "="*60)
        print("🛡️ ERROR HANDLING & EDGE CASES DEMONSTRATION")
        print("="*60)
        
        test_cases = [
            ("Empty image", np.zeros((100, 100, 3), dtype=np.uint8)),
            ("Invalid path", "non_existent_file.jpg"),
            ("Corrupted data", b"invalid_image_data"),
            ("None input", None),
        ]
        
        for test_name, test_input in test_cases:
            print(f"\n🧪 Testing: {test_name}")
            try:
                result = self.engine(test_input)
                if result and result.txts:
                    print(f"  ✓ Handled gracefully, detected: {len(result.txts)} texts")
                else:
                    print(f"  ✓ Handled gracefully, no text detected")
            except Exception as e:
                print(f"  ⚠ Exception handled: {type(e).__name__}: {e}")
    
    def run_performance_benchmark(self):
        """Run performance benchmarking"""
        print("\n" + "="*60)
        print("⚡ PERFORMANCE BENCHMARKING")
        print("="*60)
        
        if not self.performance_stats:
            print("📊 No performance data available")
            return
        
        total_time = sum(stat["processing_time"] for stat in self.performance_stats)
        avg_time = total_time / len(self.performance_stats)
        total_texts = sum(stat["text_count"] for stat in self.performance_stats)
        
        print(f"📈 Performance Statistics:")
        print(f"  📷 Images processed: {len(self.performance_stats)}")
        print(f"  ⏱ Total processing time: {total_time:.3f}s")
        print(f"  📊 Average time per image: {avg_time:.3f}s")
        print(f"  📝 Total texts detected: {total_texts}")
        print(f"  🔢 Average texts per image: {total_texts/len(self.performance_stats):.1f}")
        
        # Find fastest and slowest
        fastest = min(self.performance_stats, key=lambda x: x["processing_time"])
        slowest = max(self.performance_stats, key=lambda x: x["processing_time"])
        
        print(f"\n🏆 Performance Records:")
        print(f"  🚀 Fastest: {fastest['image']} ({fastest['processing_time']:.3f}s)")
        print(f"  🐌 Slowest: {slowest['image']} ({slowest['processing_time']:.3f}s)")
        
        # Save performance report
        perf_report_path = self.results_dir / "performance_report.json"
        with open(perf_report_path, 'w') as f:
            json.dump({
                "summary": {
                    "total_images": len(self.performance_stats),
                    "total_time": total_time,
                    "average_time": avg_time,
                    "total_texts": total_texts,
                    "fastest_image": fastest["image"],
                    "fastest_time": fastest["processing_time"],
                    "slowest_image": slowest["image"],
                    "slowest_time": slowest["processing_time"]
                },
                "detailed_stats": self.performance_stats
            }, f, indent=2)
        
        print(f"💾 Performance report saved: {perf_report_path}")
    
    def run_feature_showcase(self):
        """Showcase advanced features"""
        print("\n" + "="*60)
        print("✨ ADVANCED FEATURES SHOWCASE")
        print("="*60)
        
        # Show different output formats
        test_img = self.test_images_dir / "ch_en_num.jpg"
        if test_img.exists():
            print(f"📷 Demonstrating output formats with: {test_img.name}")
            
            try:
                result = self.engine(str(test_img))
                
                if result and result.txts:
                    print("\n📋 JSON Format:")
                    print(json.dumps(result.to_json()[:2], indent=2, ensure_ascii=False))
                    
                    print("\n📋 Markdown Format:")
                    print(result.to_markdown())
                    
                    print("\n📋 Plain Text Format:")
                    for txt in result.txts:
                        print(f"  • {txt}")
                        
            except Exception as e:
                print(f"❌ Error in feature showcase: {e}")
        
        # Show configuration options
        print(f"\n⚙️ Configuration Showcase:")
        print(f"  🔧 Engine: {'Mock' if self.use_mock else 'RapidOCR'}")
        print(f"  📁 Test images directory: {self.test_images_dir}")
        print(f"  💾 Results directory: {self.results_dir}")
        
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*60)
        print("📋 GENERATING FINAL REPORT")
        print("="*60)
        
        report = {
            "demo_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "engine_type": "Mock" if self.use_mock else "RapidOCR",
                "test_images_directory": str(self.test_images_dir),
                "results_directory": str(self.results_dir)
            },
            "capabilities_demonstrated": [
                "Basic OCR (Detection + Recognition)",
                "Component Testing (Detection/Classification/Recognition)",
                "Batch Processing",
                "Error Handling & Edge Cases",
                "Performance Benchmarking",
                "Multiple Output Formats (JSON, Markdown, Plain Text)",
                "Visualization Generation",
                "Comprehensive Reporting"
            ],
            "files_generated": list(f.name for f in self.results_dir.glob("*") if f.is_file()),
            "performance_summary": {
                "total_stats": len(self.performance_stats),
                "stats": self.performance_stats
            }
        }
        
        # Save final report
        final_report_path = self.results_dir / "final_demo_report.json"
        with open(final_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Generate markdown summary
        md_summary_path = self.results_dir / "DEMO_SUMMARY.md"
        with open(md_summary_path, 'w', encoding='utf-8') as f:
            f.write("# RapidOCR Comprehensive Demo Report\n\n")
            f.write(f"**Demo Date:** {report['demo_info']['timestamp']}\n")
            f.write(f"**Engine:** {report['demo_info']['engine_type']}\n\n")
            
            f.write("## Capabilities Demonstrated\n\n")
            for capability in report["capabilities_demonstrated"]:
                f.write(f"- ✓ {capability}\n")
            
            f.write(f"\n## Generated Files\n\n")
            for file_name in report["files_generated"]:
                f.write(f"- 📄 {file_name}\n")
            
            if self.performance_stats:
                f.write(f"\n## Performance Summary\n\n")
                total_time = sum(stat["processing_time"] for stat in self.performance_stats)
                avg_time = total_time / len(self.performance_stats)
                f.write(f"- **Images Processed:** {len(self.performance_stats)}\n")
                f.write(f"- **Total Time:** {total_time:.3f}s\n")
                f.write(f"- **Average Time per Image:** {avg_time:.3f}s\n")
        
        print(f"📄 Final report saved: {final_report_path}")
        print(f"📝 Summary saved: {md_summary_path}")
        
        return report
    
    def run_full_demo(self):
        """Run the complete demonstration"""
        print("🚀 RAPIDOCR COMPREHENSIVE FUNCTIONALITY DEMO")
        print("=" * 80)
        print("This demo showcases the complete capabilities of RapidOCR")
        print("including OCR, batch processing, error handling, and more!")
        print("=" * 80)
        
        # Run all demo components
        self.run_basic_ocr_demo()
        self.run_component_testing()
        self.run_batch_processing_demo()
        self.run_error_handling_demo()
        self.run_performance_benchmark()
        self.run_feature_showcase()
        
        # Generate final report
        report = self.generate_final_report()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 All results saved to: {self.results_dir}")
        print(f"📄 Check final report: {self.results_dir}/final_demo_report.json")
        print(f"📝 Check summary: {self.results_dir}/DEMO_SUMMARY.md")
        print("\nDemo showcased:")
        for capability in report["capabilities_demonstrated"]:
            print(f"  ✓ {capability}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="RapidOCR Comprehensive Demo - Full Functionality Showcase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_comprehensive.py                    # Run full demo
  python demo_comprehensive.py --mock             # Force mock mode
  python demo_comprehensive.py --basic-only       # Run basic OCR only
  python demo_comprehensive.py --batch-only       # Run batch processing only
        """
    )
    
    parser.add_argument("--mock", action="store_true", 
                       help="Force mock mode (useful for offline demonstration)")
    parser.add_argument("--basic-only", action="store_true",
                       help="Run only basic OCR demonstration")
    parser.add_argument("--batch-only", action="store_true", 
                       help="Run only batch processing demonstration")
    parser.add_argument("--performance-only", action="store_true",
                       help="Run only performance benchmarking")
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = RapidOCRDemo(use_mock=args.mock)
    
    try:
        if args.basic_only:
            demo.run_basic_ocr_demo()
        elif args.batch_only:
            demo.run_batch_processing_demo()
        elif args.performance_only:
            demo.run_performance_benchmark()
        else:
            demo.run_full_demo()
            
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()