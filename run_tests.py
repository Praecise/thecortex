#!/usr/bin/env python3
import subprocess
import argparse
import sys
import os
from datetime import datetime

def run_tests(component=None, coverage=False, verbose=False):
    """Run test suite with specified options"""
    # Base command
    cmd = ['pytest']
    
    # Add verbosity
    if verbose:
        cmd.append('-v')
    
    # Add coverage reporting
    if coverage:
        cmd.extend(['--cov=cortex', '--cov-report=html', '--cov-report=term'])
    
    # Component specific tests
    if component:
        if component == 'all':
            cmd.append('tests/')
        else:
            cmd.append(f'tests/test_{component}.py')
    else:
        cmd.append('tests/')
    
    # Create test results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Run tests and capture output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'test_results/test_run_{timestamp}.log'
    
    print(f"Running tests: {' '.join(cmd)}")
    print(f"Logging output to: {log_file}")
    
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Write output to log file
        f.write(result.stdout.decode())
        f.write('\n\nSTDERR:\n')
        f.write(result.stderr.decode())
    
    # Print output to console
    print(result.stdout.decode())
    if result.stderr:
        print("\nErrors/Warnings:", file=sys.stderr)
        print(result.stderr.decode(), file=sys.stderr)
    
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description='Run Cortex test suite')
    parser.add_argument('--component', choices=['all', 'model', 'training', 'network', 'security', 'integration'],
                      help='Specific component to test')
    parser.add_argument('--coverage', action='store_true',
                      help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Verbose output')
    
    args = parser.parse_args()
    
    # Run pre-test setup
    print("Setting up test environment...")
    os.makedirs('logs', exist_ok=True)
    
    # Run tests
    return run_tests(args.component, args.coverage, args.verbose)

if __name__ == '__main__':
    sys.exit(main())