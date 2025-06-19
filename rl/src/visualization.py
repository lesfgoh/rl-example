import os
import time
import sys
import pygame
import numpy as np
from PIL import Image
from typing import Optional, Union


class VisualizationHandler:
    """
    Handles visualization and interactive controls for the GridWorld environment.
    Provides a reusable way to visualize environment states and handle keyboard controls.
    """
    
    def __init__(self, render_mode: Optional[str] = "human"):
        """
        Initialize the visualization handler
        
        Args:
            render_mode: The render mode to use ("human" or None)
        """
        self.render_mode = render_mode
        self.interactive_mode = render_mode == "human"
        self.frame_count = 0
        
        # Default control settings
        self.paused = True
        self.step_mode = True
        
        # Only initialize pygame if we're in interactive mode
        if self.interactive_mode:
            pygame.init()
            print("Started in step mode. Press SPACE to advance or 's' to disable step mode.")
    
    def handle_events(self, env):
        """
        Handle pygame events for visualization control
        
        Args:
            env: The environment object with render and close methods
            
        Returns:
            bool: True if execution should continue, False if it should terminate
        """
        if not self.interactive_mode:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                sys.exit()
                return False
            elif event.type == pygame.KEYDOWN:
                # Press 'p' to toggle pause
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                    print(f"Game {'paused' if self.paused else 'resumed'}")
                # Press 's' to toggle step mode
                elif event.key == pygame.K_s:
                    self.step_mode = not self.step_mode
                    if not self.step_mode:
                        self.paused = False
                    print(f"Step mode {'enabled' if self.step_mode else 'disabled'}")
                # Press space to advance one step in step mode
                elif event.key == pygame.K_SPACE and self.step_mode:
                    # Just unpause for one iteration
                    self.paused = False
                    print("Advancing one step")
                # Press 'q' to quit
                elif event.key == pygame.K_q:
                    env.close()
                    pygame.quit()
                    sys.exit()
                    return False
        
        return True
    
    def handle_pause(self, env):
        """
        Handle paused state by waiting for user input
        
        Args:
            env: The environment object with render method
            
        Returns:
            bool: True if execution should continue, False if it should terminate
        """
        if not self.interactive_mode or not self.paused:
            return True
            
        while self.paused:
            if env.render_mode == "human":
                env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit()
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                        print(f"Game {'paused' if self.paused else 'resumed'}")
                    elif event.key == pygame.K_SPACE and self.step_mode:
                        # Unpause for one iteration
                        self.paused = False
                        print("Advancing one step")
                        break  # Exit the event loop to advance the step
                    elif event.key == pygame.K_s:
                        self.step_mode = not self.step_mode
                        if not self.step_mode:
                            self.paused = False
                        print(f"Step mode {'enabled' if self.step_mode else 'disabled'}")
                    elif event.key == pygame.K_q:
                        env.close()
                        pygame.quit()
                        sys.exit()
                        return False
            
            if not self.paused:
                break
                
            time.sleep(0.1)
        
        return True
    
    def save_frame(self, env):
        """
        Save a frame from the environment if in rgb_array mode
        
        Args:
            env: The environment object with render method
        """
        if env.render_mode == "rgb_array":
            frame = env.render()
            if frame is not None and isinstance(frame, np.ndarray):
                img = Image.fromarray(frame)
                os.makedirs("frames", exist_ok=True)
                img.save(f"frames/frame_{self.frame_count:06d}.png")
                self.frame_count += 1
    
    def auto_pause_if_needed(self):
        """
        Auto-pause if in step mode
        """
        if self.interactive_mode and self.step_mode:
            self.paused = True
    
    def shutdown(self, env):
        """
        Clean up visualization resources
        
        Args:
            env: The environment object with close method
        """
        env.close()
        if self.interactive_mode:
            pygame.quit() 