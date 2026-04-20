'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Switch } from '@/components/ui/switch';
import { getWorkspace, updateWorkspace } from '@/lib/api/edgequake';
import { useTenantStore } from '@/stores/use-tenant-store';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { GitBranch, Pencil, Save, X } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';

/**
 * Workspace-level controls for the Reference-Repo detection pipeline.
 *
 * The single toggle here feeds the task-processor's post-verification
 * filter (`accept_unofficial_implementations` in the backend). When off
 * (default), candidates the LLM verifier classified as `third_party` or
 * `unrelated` are dropped before they ever hit the review queue — keeping
 * the References tab focused on author-released repos. When on, the
 * verdict is still attached to each row but nothing is dropped.
 */
export function ReferenceRepoSettingsCard() {
  const queryClient = useQueryClient();
  const { selectedTenantId, selectedWorkspaceId } = useTenantStore();

  const [isEditing, setIsEditing] = useState(false);
  const [acceptUnofficial, setAcceptUnofficial] = useState(false);

  const { data: workspace, isLoading } = useQuery({
    queryKey: ['workspace', selectedTenantId, selectedWorkspaceId],
    queryFn: () => getWorkspace(selectedTenantId!, selectedWorkspaceId!),
    enabled: !!selectedTenantId && !!selectedWorkspaceId,
    staleTime: 60000,
    retry: 1,
  });

  const updateMutation = useMutation({
    mutationFn: () =>
      updateWorkspace(selectedTenantId!, selectedWorkspaceId!, {
        accept_unofficial_implementations: acceptUnofficial,
      }),
    onSuccess: () => {
      toast.success('Reference-repo detection settings updated');
      queryClient.invalidateQueries({
        queryKey: ['workspace', selectedTenantId, selectedWorkspaceId],
      });
      setIsEditing(false);
    },
    onError: (error) => {
      toast.error('Failed to update settings', {
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    },
  });

  const handleEdit = () => {
    setAcceptUnofficial(
      workspace?.accept_unofficial_implementations ?? false,
    );
    setIsEditing(true);
  };

  if (!selectedTenantId || !selectedWorkspaceId) {
    return null;
  }

  const current = workspace?.accept_unofficial_implementations ?? false;

  return (
    <Card>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-violet-600" />
            <CardTitle>Reference Repositories</CardTitle>
          </div>
          {!isEditing && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleEdit}
              aria-label="Edit"
            >
              <Pencil className="h-4 w-4" />
            </Button>
          )}
        </div>
        <CardDescription>
          Control which repo candidates the detection pipeline surfaces for
          review.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {isLoading ? (
          <Skeleton className="h-20 w-full" />
        ) : isEditing ? (
          <>
            <div className="flex items-center justify-between gap-4">
              <div className="flex-1">
                <label className="text-sm font-medium">
                  Accept unofficial implementations
                </label>
                <p className="text-xs text-muted-foreground mt-0.5">
                  When enabled, third-party and unrelated candidates are
                  still added to the review queue (with a verdict badge).
                  When disabled (default), only candidates likely authored
                  by the paper&apos;s authors appear for review.
                </p>
              </div>
              <Switch
                checked={acceptUnofficial}
                onCheckedChange={setAcceptUnofficial}
              />
            </div>
            <div className="flex items-center gap-2 pt-2">
              <Button
                size="sm"
                onClick={() => updateMutation.mutate()}
                disabled={updateMutation.isPending}
              >
                <Save className="h-4 w-4 mr-2" />
                Save
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsEditing(false)}
                disabled={updateMutation.isPending}
              >
                <X className="h-4 w-4 mr-2" />
                Cancel
              </Button>
            </div>
          </>
        ) : (
          <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
            <div className="flex-1">
              <div className="text-xs text-muted-foreground mb-0.5">
                Accept unofficial implementations
              </div>
              <div className="font-medium">{current ? 'Enabled' : 'Disabled'}</div>
              <div className="text-sm text-muted-foreground">
                {current
                  ? 'Third-party / unrelated repos appear in review queue with a verdict badge.'
                  : 'Only author-released (official) repos reach the review queue.'}
              </div>
            </div>
            <Badge variant={current ? 'default' : 'secondary'}>
              {current ? 'on' : 'off'}
            </Badge>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
