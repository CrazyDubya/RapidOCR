#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
RapidOCR CLI Utility Tool
========================

Enhanced command-line interface for RapidOCR with advanced features:
- Interactive mode
- Batch processing
- Multiple output formats
- Configuration management
- Performance monitoring
- Plugin system

Author: CLI Enhancement for RapidOCR
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob

import cv2
import numpy as np

# Add local imports
sys.path.append(str(Path(__file__).parent))
from demo_comprehensive import RapidOCRDemo
from rapidocr_optimizations import OptimizedRapidOCR


class RapidOCRCLI:
    """Enhanced CLI interface for RapidOCR"""
    
    def __init__(self):
        self.demo = None
        self.config = {
            "output_format": "json",
            "save_visualization": False,
            "batch_size": 4,
            "performance_monitoring": False,
            "verbose": False
        }
        self.results = []
        
    def initialize_engine(self, use_mock=False, **kwargs):
        """Initialize the OCR engine"""
        try:
            if use_mock:
                from demo_comprehensive import MockRapidOCR
                self.engine = MockRapidOCR()
                print("🔧 Initialized mock OCR engine")
            else:
                self.engine = OptimizedRapidOCR(**kwargs)
                print("🔧 Initialized optimized RapidOCR engine")
        except Exception as e:
            print(f"⚠ Failed to initialize engine: {e}")
            print("🔧 Falling back to mock engine")
            from demo_comprehensive import MockRapidOCR
            self.engine = MockRapidOCR()
    
    def process_single_image(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Process a single image"""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if self.config["verbose"]:
            print(f"📷 Processing: {image_path}")
        
        start_time = time.time()
        try:
            result = self.engine(image_path, **kwargs)
            processing_time = time.time() - start_time
            
            # Create result dictionary
            result_dict = {
                "filename": Path(image_path).name,
                "processing_time": processing_time,
                "success": True,
                "texts": result.txts if result and result.txts else [],
                "confidence_scores": result.scores if result and result.scores else [],
                "bounding_boxes": result.boxes if result and result.boxes else [],
                "text_count": len(result) if result else 0
            }
            
            # Save visualization if requested
            if self.config["save_visualization"] and result:
                vis_path = Path(image_path).parent / f"{Path(image_path).stem}_vis.png"
                result.vis(str(vis_path))
                result_dict["visualization_saved"] = str(vis_path)
            
            if self.config["verbose"]:
                print(f"  ✅ Processed in {processing_time:.3f}s, found {result_dict['text_count']} texts")
            
            return result_dict
            
        except Exception as e:
            processing_time = time.time() - start_time
            if self.config["verbose"]:
                print(f"  ❌ Error: {e}")
            
            return {
                "filename": Path(image_path).name,
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
    
    def process_batch(self, image_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple images in batch"""
        print(f"📚 Processing {len(image_paths)} images in batch...")
        
        results = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths, 1):
            if self.config["verbose"]:
                print(f"[{i}/{len(image_paths)}] Processing {Path(image_path).name}")
            
            result = self.process_single_image(image_path, **kwargs)
            results.append(result)
            
            # Progress indicator for non-verbose mode
            if not self.config["verbose"] and i % 10 == 0:
                print(f"  Progress: {i}/{len(image_paths)} images processed")
        
        total_time = time.time() - start_time
        successful = len([r for r in results if r.get("success", False)])
        
        print(f"📊 Batch processing completed:")
        print(f"  ✅ Successful: {successful}/{len(image_paths)}")
        print(f"  ⏱ Total time: {total_time:.2f}s")
        print(f"  📈 Average time per image: {total_time/len(image_paths):.3f}s")
        
        return results
    
    def process_directory(self, directory: str, recursive: bool = False, 
                         file_patterns: List[str] = None) -> List[Dict[str, Any]]:
        """Process all images in a directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Default file patterns
        if file_patterns is None:
            file_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        
        # Find all image files
        image_files = []
        for pattern in file_patterns:
            if recursive:
                image_files.extend(dir_path.rglob(pattern))
            else:
                image_files.extend(dir_path.glob(pattern))
        
        image_files = [str(f) for f in image_files if f.is_file()]
        
        if not image_files:
            print(f"⚠ No image files found in {directory}")
            return []
        
        print(f"📁 Found {len(image_files)} images in {directory}")
        return self.process_batch(image_files)
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config["output_format"] == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        elif self.config["output_format"] == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
        
        elif self.config["output_format"] == "markdown":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# OCR Results\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"## Image {i}: {result.get('filename', 'Unknown')}\n\n")
                    if result.get("success", False):
                        f.write(f"- **Processing Time:** {result.get('processing_time', 0):.3f}s\n")
                        f.write(f"- **Text Count:** {result.get('text_count', 0)}\n")
                        if result.get("texts"):
                            f.write("- **Detected Text:**\n")
                            for text in result["texts"]:
                                f.write(f"  - {text}\n")
                    else:
                        f.write(f"- **Error:** {result.get('error', 'Unknown error')}\n")
                    f.write("\n")
        
        print(f"💾 Results saved to: {output_path}")
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary report from results"""
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        if successful_results:
            processing_times = [r["processing_time"] for r in successful_results]
            text_counts = [r["text_count"] for r in successful_results]
            
            summary = {
                "total_images": len(results),
                "successful_images": len(successful_results),
                "failed_images": len(failed_results),
                "success_rate": len(successful_results) / len(results) if results else 0,
                "total_processing_time": sum(processing_times),
                "average_processing_time": sum(processing_times) / len(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "total_texts_detected": sum(text_counts),
                "average_texts_per_image": sum(text_counts) / len(text_counts)
            }
        else:
            summary = {
                "total_images": len(results),
                "successful_images": 0,
                "failed_images": len(results),
                "success_rate": 0,
                "error_summary": [r.get("error", "Unknown") for r in failed_results]
            }
        
        return summary
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("🔍 RapidOCR Interactive Mode")
        print("Type 'help' for available commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\nrapidocr> ").strip()
                
                if command == "quit" or command == "exit":
                    print("👋 Goodbye!")
                    break
                
                elif command == "help":
                    self.show_interactive_help()
                
                elif command.startswith("process "):
                    image_path = command[8:].strip()
                    if image_path:
                        try:
                            result = self.process_single_image(image_path)
                            self.print_result(result)
                        except Exception as e:
                            print(f"❌ Error: {e}")
                    else:
                        print("❌ Please specify an image path")
                
                elif command.startswith("batch "):
                    directory = command[6:].strip()
                    if directory:
                        try:
                            results = self.process_directory(directory)
                            summary = self.generate_summary_report(results)
                            print(f"\n📊 Summary: {summary['successful_images']}/{summary['total_images']} successful")
                        except Exception as e:
                            print(f"❌ Error: {e}")
                    else:
                        print("❌ Please specify a directory path")
                
                elif command.startswith("config "):
                    config_cmd = command[7:].strip()
                    self.handle_config_command(config_cmd)
                
                elif command == "status":
                    self.show_status()
                
                elif command == "clear":
                    os.system('clear' if os.name == 'posix' else 'cls')
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_interactive_help(self):
        """Show help for interactive mode"""
        help_text = """
Available Commands:
  process <image_path>     - Process a single image
  batch <directory>        - Process all images in directory
  config <setting> <value> - Change configuration
  status                   - Show current status
  clear                    - Clear screen
  help                     - Show this help
  quit/exit               - Exit interactive mode

Configuration Options:
  output_format: json, csv, markdown
  save_visualization: true, false
  verbose: true, false

Examples:
  process image.jpg
  batch ./images
  config output_format markdown
  config verbose true
        """
        print(help_text)
    
    def handle_config_command(self, config_cmd: str):
        """Handle configuration commands"""
        parts = config_cmd.split()
        if len(parts) != 2:
            print("❌ Usage: config <setting> <value>")
            return
        
        setting, value = parts
        if setting in self.config:
            # Convert string values to appropriate types
            if value.lower() in ["true", "false"]:
                self.config[setting] = value.lower() == "true"
            elif value.isdigit():
                self.config[setting] = int(value)
            else:
                self.config[setting] = value
            
            print(f"✅ Set {setting} = {self.config[setting]}")
        else:
            print(f"❌ Unknown setting: {setting}")
            print(f"Available settings: {list(self.config.keys())}")
    
    def show_status(self):
        """Show current status"""
        print(f"🔧 Engine: {'Initialized' if hasattr(self, 'engine') else 'Not initialized'}")
        print(f"⚙️ Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
    
    def print_result(self, result: Dict[str, Any]):
        """Print formatted result"""
        if result.get("success", False):
            print(f"✅ {result['filename']}")
            print(f"  ⏱ Processing time: {result['processing_time']:.3f}s")
            print(f"  📝 Text count: {result['text_count']}")
            if result.get("texts"):
                print("  📋 Detected text:")
                for text in result["texts"]:
                    print(f"    • {text}")
        else:
            print(f"❌ {result['filename']}: {result.get('error', 'Unknown error')}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="RapidOCR Enhanced CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rapidocr_cli.py image.jpg                           # Process single image
  rapidocr_cli.py images/ --batch                     # Process directory
  rapidocr_cli.py image.jpg --output results.json    # Save to file
  rapidocr_cli.py --interactive                       # Interactive mode
  rapidocr_cli.py images/ --batch --recursive         # Recursive directory
        """
    )
    
    # Input options
    parser.add_argument("input", nargs="?", help="Input image file or directory")
    parser.add_argument("--batch", action="store_true", help="Process directory in batch mode")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    # Output options
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", choices=["json", "csv", "markdown"], default="json",
                       help="Output format (default: json)")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images")
    
    # OCR options
    parser.add_argument("--mock", action="store_true", help="Use mock engine")
    parser.add_argument("--text-score", type=float, default=0.5, help="Text detection threshold")
    parser.add_argument("--no-det", action="store_true", help="Disable text detection")
    parser.add_argument("--no-cls", action="store_true", help="Disable text classification")
    parser.add_argument("--no-rec", action="store_true", help="Disable text recognition")
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = RapidOCRCLI()
    
    # Set configuration
    cli.config["output_format"] = args.format
    cli.config["save_visualization"] = args.save_vis
    cli.config["verbose"] = args.verbose and not args.quiet
    
    # Initialize engine
    ocr_params = {
        "text_score": args.text_score,
        "use_det": not args.no_det,
        "use_cls": not args.no_cls,
        "use_rec": not args.no_rec
    }
    cli.initialize_engine(use_mock=args.mock, **ocr_params)
    
    try:
        if args.interactive:
            # Interactive mode
            cli.interactive_mode()
        
        elif args.input:
            input_path = Path(args.input)
            
            if input_path.is_file():
                # Process single file
                result = cli.process_single_image(str(input_path), **ocr_params)
                results = [result]
                
                if not args.quiet:
                    cli.print_result(result)
            
            elif input_path.is_dir() and args.batch:
                # Process directory
                results = cli.process_directory(str(input_path), recursive=args.recursive)
                
                if not args.quiet:
                    summary = cli.generate_summary_report(results)
                    print(f"\n📊 Summary Report:")
                    for key, value in summary.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.3f}")
                        else:
                            print(f"  {key}: {value}")
            
            else:
                print(f"❌ Invalid input: {args.input}")
                print("Use --batch flag for directory processing")
                return 1
            
            # Save results if output specified
            if args.output and results:
                cli.save_results(results, args.output)
        
        else:
            # No input specified, show help
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())