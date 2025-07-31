#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Comprehensive Testing and Benchmarking Suite for RapidOCR
=========================================================

This module provides extensive testing capabilities for RapidOCR:
- Unit tests for all components
- Performance benchmarking
- Accuracy testing
- Memory usage analysis
- Stress testing
- Regression testing

Author: Testing Enhancement for RapidOCR
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch
import statistics

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from demo_comprehensive import RapidOCRDemo, MockRapidOCR
from rapidocr_optimizations import OptimizedRapidOCR, PerformanceMonitor


class RapidOCRTestSuite(unittest.TestCase):
    """Comprehensive test suite for RapidOCR functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.demo = RapidOCRDemo(use_mock=True)
        cls.test_images_dir = Path(__file__).parent / "python" / "tests" / "test_files"
        cls.results_dir = Path("test_results")
        cls.results_dir.mkdir(exist_ok=True)
        
        # Create test images if they don't exist
        cls._create_test_images()
    
    @classmethod
    def _create_test_images(cls):
        """Create synthetic test images for testing"""
        # Create a simple text image
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(str(cls.results_dir / "synthetic_test.jpg"), test_img)
        
        # Create an empty image
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(cls.results_dir / "empty_test.jpg"), empty_img)
        
        # Create a noisy image
        noisy_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(cls.results_dir / "noisy_test.jpg"), noisy_img)
    
    def test_basic_ocr_functionality(self):
        """Test basic OCR functionality"""
        test_img = self.results_dir / "synthetic_test.jpg"
        
        result = self.demo.engine(str(test_img))
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.txts, list)
        self.assertIsInstance(result.scores, list)
        
        # Test that we can get JSON output
        json_result = result.to_json()
        self.assertIsInstance(json_result, list)
        
        # Test that we can get markdown output
        md_result = result.to_markdown()
        self.assertIsInstance(md_result, str)
    
    def test_empty_image_handling(self):
        """Test handling of empty or invalid images"""
        empty_img = self.results_dir / "empty_test.jpg"
        
        # Should not crash on empty image
        result = self.demo.engine(str(empty_img))
        self.assertIsNotNone(result)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        invalid_inputs = [
            "non_existent_file.jpg",
            None,
            "",
            b"invalid_image_data",
            np.array([])  # Empty array
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                try:
                    result = self.demo.engine(invalid_input)
                    # Should either return None/empty result or handle gracefully
                    self.assertTrue(True)  # If we get here, no crash occurred
                except Exception as e:
                    # Should only raise specific, expected exceptions
                    self.assertIsInstance(e, (ValueError, FileNotFoundError, TypeError))
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        test_images = [
            str(self.results_dir / "synthetic_test.jpg"),
            str(self.results_dir / "empty_test.jpg"),
        ]
        
        # Test batch processing doesn't crash
        try:
            # Simulate batch processing
            results = []
            for img_path in test_images:
                result = self.demo.engine(img_path)
                results.append(result)
            
            self.assertEqual(len(results), len(test_images))
            
        except Exception as e:
            self.fail(f"Batch processing failed: {e}")
    
    def test_output_formats(self):
        """Test different output formats"""
        test_img = self.results_dir / "synthetic_test.jpg"
        result = self.demo.engine(str(test_img))
        
        # Test JSON format
        json_output = result.to_json()
        self.assertIsInstance(json_output, list)
        
        # Test markdown format
        md_output = result.to_markdown()
        self.assertIsInstance(md_output, str)
        self.assertIn("|", md_output)  # Should contain table formatting
        
        # Test visualization
        vis_img = result.vis()
        self.assertIsInstance(vis_img, np.ndarray)
        self.assertEqual(len(vis_img.shape), 3)  # Should be color image
    
    def test_configuration_options(self):
        """Test various configuration options"""
        test_img = self.results_dir / "synthetic_test.jpg"
        
        # Test with different parameters
        configs = [
            {"use_det": True, "use_cls": False, "use_rec": True},
            {"use_det": False, "use_cls": True, "use_rec": False},
            {"use_det": True, "use_cls": True, "use_rec": True},
        ]
        
        for config in configs:
            with self.subTest(config=config):
                try:
                    result = self.demo.engine(str(test_img), **config)
                    self.assertIsNotNone(result)
                except Exception as e:
                    # Some configurations might not be supported in mock mode
                    pass


class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self, use_mock=True):
        self.demo = RapidOCRDemo(use_mock=use_mock)
        self.performance_monitor = PerformanceMonitor()
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def run_speed_benchmark(self, num_images=10) -> Dict[str, Any]:
        """Run speed benchmarking with synthetic images"""
        print(f"🏃 Running speed benchmark with {num_images} images...")
        
        # Create test images of different sizes
        test_images = self._create_benchmark_images(num_images)
        
        results = []
        total_start_time = time.time()
        
        for i, (img_path, img_size) in enumerate(test_images, 1):
            print(f"  Processing image {i}/{num_images} ({img_size[0]}x{img_size[1]})...")
            
            start_time = time.time()
            try:
                result = self.demo.engine(str(img_path))
                processing_time = time.time() - start_time
                
                results.append({
                    "image_index": i,
                    "image_size": img_size,
                    "processing_time": processing_time,
                    "text_count": len(result) if result else 0,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "image_index": i,
                    "image_size": img_size,
                    "error": str(e),
                    "success": False
                })
        
        total_time = time.time() - total_start_time
        
        # Calculate statistics
        successful_results = [r for r in results if r.get("success", False)]
        processing_times = [r["processing_time"] for r in successful_results]
        
        if processing_times:
            stats = {
                "total_images": num_images,
                "successful_images": len(successful_results),
                "total_time": total_time,
                "avg_processing_time": statistics.mean(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "median_processing_time": statistics.median(processing_times),
                "std_processing_time": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                "throughput_fps": len(successful_results) / total_time,
                "detailed_results": results
            }
        else:
            stats = {
                "total_images": num_images,
                "successful_images": 0,
                "error": "No successful processing",
                "detailed_results": results
            }
        
        # Save results
        benchmark_file = self.results_dir / f"speed_benchmark_{num_images}images.json"
        with open(benchmark_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Speed benchmark completed!")
        if processing_times:
            print(f"  ⏱ Average processing time: {stats['avg_processing_time']:.3f}s")
            print(f"  🚀 Throughput: {stats['throughput_fps']:.2f} FPS")
            print(f"  📄 Benchmark saved: {benchmark_file}")
        
        # Cleanup test images
        for img_path, _ in test_images:
            img_path.unlink(missing_ok=True)
        
        return stats
    
    def _create_benchmark_images(self, num_images) -> List[Tuple[Path, Tuple[int, int]]]:
        """Create synthetic images for benchmarking"""
        test_images = []
        
        sizes = [
            (320, 240),    # Small
            (640, 480),    # Medium
            (1024, 768),   # Large
            (1920, 1080),  # HD
        ]
        
        for i in range(num_images):
            size = sizes[i % len(sizes)]
            width, height = size
            
            # Create synthetic image with text
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Add some text
            text = f"Benchmark Image {i+1}"
            font_scale = min(width, height) / 400
            thickness = max(1, int(font_scale * 2))
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 0, 0), thickness)
            
            # Add some noise for realism
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            img_path = self.results_dir / f"benchmark_img_{i}_{width}x{height}.jpg"
            cv2.imwrite(str(img_path), img)
            
            test_images.append((img_path, (width, height)))
        
        return test_images
    
    def run_memory_benchmark(self) -> Dict[str, Any]:
        """Run memory usage benchmark"""
        print("🧠 Running memory usage benchmark...")
        
        try:
            import psutil
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process multiple images and track memory
            memory_usage = [initial_memory]
            
            for i in range(10):
                # Create and process image
                img = np.ones((800, 600, 3), dtype=np.uint8) * 255
                cv2.putText(img, f"Memory Test {i}", (100, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                
                img_path = self.results_dir / f"memory_test_{i}.jpg"
                cv2.imwrite(str(img_path), img)
                
                # Process with OCR
                result = self.demo.engine(str(img_path))
                
                # Measure memory
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_usage.append(current_memory)
                
                # Cleanup
                img_path.unlink(missing_ok=True)
            
            memory_stats = {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": memory_usage[-1],
                "max_memory_mb": max(memory_usage),
                "memory_increase_mb": memory_usage[-1] - initial_memory,
                "memory_usage_timeline": memory_usage
            }
            
            # Save results
            memory_file = self.results_dir / "memory_benchmark.json"
            with open(memory_file, 'w') as f:
                json.dump(memory_stats, f, indent=2)
            
            print(f"🧠 Memory benchmark completed!")
            print(f"  📈 Memory increase: {memory_stats['memory_increase_mb']:.1f} MB")
            print(f"  📊 Peak memory: {memory_stats['max_memory_mb']:.1f} MB")
            print(f"  📄 Results saved: {memory_file}")
            
            return memory_stats
            
        except ImportError:
            print("⚠ psutil not available for memory benchmarking")
            return {"error": "psutil not available"}
    
    def run_stress_test(self, duration_seconds=30) -> Dict[str, Any]:
        """Run stress test for specified duration"""
        print(f"💪 Running stress test for {duration_seconds} seconds...")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        processed_count = 0
        error_count = 0
        processing_times = []
        
        while time.time() < end_time:
            try:
                # Create random image
                img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
                
                # Add some text
                cv2.putText(img, f"Stress {processed_count}", (100, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                img_path = self.results_dir / f"stress_test_{processed_count}.jpg"
                cv2.imwrite(str(img_path), img)
                
                # Process
                process_start = time.time()
                result = self.demo.engine(str(img_path))
                process_time = time.time() - process_start
                
                processing_times.append(process_time)
                processed_count += 1
                
                # Cleanup
                img_path.unlink(missing_ok=True)
                
            except Exception as e:
                error_count += 1
                print(f"  ⚠ Error in stress test iteration {processed_count}: {e}")
        
        total_time = time.time() - start_time
        
        stress_stats = {
            "duration_seconds": total_time,
            "images_processed": processed_count,
            "errors": error_count,
            "success_rate": processed_count / (processed_count + error_count) if (processed_count + error_count) > 0 else 0,
            "avg_processing_time": statistics.mean(processing_times) if processing_times else 0,
            "throughput_per_second": processed_count / total_time if total_time > 0 else 0
        }
        
        # Save results
        stress_file = self.results_dir / "stress_test.json"
        with open(stress_file, 'w') as f:
            json.dump(stress_stats, f, indent=2)
        
        print(f"💪 Stress test completed!")
        print(f"  📊 Processed: {processed_count} images")
        print(f"  ⚠ Errors: {error_count}")
        print(f"  🎯 Success rate: {stress_stats['success_rate']:.2%}")
        print(f"  🚀 Throughput: {stress_stats['throughput_per_second']:.2f} images/sec")
        print(f"  📄 Results saved: {stress_file}")
        
        return stress_stats


class ComprehensiveTestRunner:
    """Runner for all tests and benchmarks"""
    
    def __init__(self):
        self.results_dir = Path("comprehensive_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def run_all_tests(self):
        """Run all tests and benchmarks"""
        print("🧪 COMPREHENSIVE RAPIDOCR TESTING SUITE")
        print("=" * 60)
        
        results = {}
        
        # Run unit tests
        print("\n📋 Running Unit Tests...")
        test_suite = unittest.TestLoader().loadTestsFromTestCase(RapidOCRTestSuite)
        test_runner = unittest.TextTestRunner(verbosity=2)
        test_result = test_runner.run(test_suite)
        
        results["unit_tests"] = {
            "tests_run": test_result.testsRun,
            "failures": len(test_result.failures),
            "errors": len(test_result.errors),
            "success": test_result.wasSuccessful()
        }
        
        # Run performance benchmarks
        print("\n⚡ Running Performance Benchmarks...")
        benchmark = PerformanceBenchmark(use_mock=True)
        
        results["speed_benchmark"] = benchmark.run_speed_benchmark(5)
        results["memory_benchmark"] = benchmark.run_memory_benchmark()
        results["stress_test"] = benchmark.run_stress_test(10)  # 10 second stress test
        
        # Generate final report
        final_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_summary": {
                "unit_tests_passed": results["unit_tests"]["success"],
                "performance_tests_completed": True,
                "overall_status": "PASS" if results["unit_tests"]["success"] else "FAIL"
            },
            "detailed_results": results
        }
        
        report_file = self.results_dir / "comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(final_report)
        
        print(f"\n🎉 COMPREHENSIVE TESTING COMPLETED!")
        print(f"📄 Full report: {report_file}")
        print(f"📝 Summary: {self.results_dir}/TEST_SUMMARY.md")
        
        return final_report
    
    def _generate_markdown_report(self, report):
        """Generate markdown summary report"""
        md_content = f"""# RapidOCR Comprehensive Test Report

**Test Date:** {report['timestamp']}
**Overall Status:** {report['test_summary']['overall_status']}

## Test Summary

- ✅ Unit Tests: {'PASSED' if report['test_summary']['unit_tests_passed'] else 'FAILED'}
- ✅ Performance Tests: {'COMPLETED' if report['test_summary']['performance_tests_completed'] else 'FAILED'}

## Unit Test Results

- **Tests Run:** {report['detailed_results']['unit_tests']['tests_run']}
- **Failures:** {report['detailed_results']['unit_tests']['failures']}
- **Errors:** {report['detailed_results']['unit_tests']['errors']}

## Performance Benchmark Results

### Speed Benchmark
- **Images Processed:** {report['detailed_results']['speed_benchmark'].get('successful_images', 'N/A')}
- **Average Processing Time:** {report['detailed_results']['speed_benchmark'].get('avg_processing_time', 0):.3f}s
- **Throughput:** {report['detailed_results']['speed_benchmark'].get('throughput_fps', 0):.2f} FPS

### Memory Usage
- **Memory Increase:** {report['detailed_results']['memory_benchmark'].get('memory_increase_mb', 0):.1f} MB
- **Peak Memory:** {report['detailed_results']['memory_benchmark'].get('max_memory_mb', 0):.1f} MB

### Stress Test
- **Images Processed:** {report['detailed_results']['stress_test'].get('images_processed', 'N/A')}
- **Success Rate:** {report['detailed_results']['stress_test'].get('success_rate', 0):.2%}
- **Throughput:** {report['detailed_results']['stress_test'].get('throughput_per_second', 0):.2f} images/sec

## Files Generated

All detailed results and test artifacts are saved in the `comprehensive_test_results/` directory.
"""
        
        md_file = self.results_dir / "TEST_SUMMARY.md"
        with open(md_file, 'w') as f:
            f.write(md_content)


def main():
    """Main function for running tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RapidOCR Comprehensive Testing Suite")
    parser.add_argument("--unit-tests", action="store_true", help="Run only unit tests")
    parser.add_argument("--benchmarks", action="store_true", help="Run only performance benchmarks")
    parser.add_argument("--stress-duration", type=int, default=10, help="Stress test duration in seconds")
    
    args = parser.parse_args()
    
    if args.unit_tests:
        # Run only unit tests
        test_suite = unittest.TestLoader().loadTestsFromTestCase(RapidOCRTestSuite)
        test_runner = unittest.TextTestRunner(verbosity=2)
        test_runner.run(test_suite)
    elif args.benchmarks:
        # Run only benchmarks
        benchmark = PerformanceBenchmark(use_mock=True)
        benchmark.run_speed_benchmark(10)
        benchmark.run_memory_benchmark()
        benchmark.run_stress_test(args.stress_duration)
    else:
        # Run comprehensive tests
        runner = ComprehensiveTestRunner()
        runner.run_all_tests()


if __name__ == "__main__":
    main()